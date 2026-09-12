/*==================================================================================================
* Project : BackPropagation
* Platform : S32K3XX
* Author: Iván Delgado Ramos
* Date: 11/09/26
* Copyright 2026
*
==================================================================================================*/

/**
* Implementación de red neuronal multicapa
*   1 -- 2 -- 3 -- 1
*   por activación de sigmoide parametrizado
*   utilizando el algoritmo back-forward
*   (clasificacion binaria: 0 = mayuscula, 1 = minuscula)
*/

/**         HEADERS        **/
#include "Clock_Ip.h"
#include "Siul2_Port_Ip.h"
#include "Lpuart_Uart_Ip.h"
#include "IntCtrl_Ip.h"
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "Lpuart_Uart_Ip_Irq.h"
#include <math.h>
#include "pesos_iniciales.h"


#ifdef __cplusplus
extern "C"{
#endif

/*Definitiones utiles (perfifericos y debug)*/
#define SIUL2_CONFIG g_pin_mux_InitConfigArr_PortContainer_0_BOARD_InitPeripherals
#define SIUL2_PINS NUM_OF_CONFIGURED_PINS_PortContainer_0_BOARD_InitPeripherals
#define UART_INSTANCE 6
#define BUFFER_SIZE 256U
#define DWT_LAR     (*(volatile uint32 *)0xE0001FB0UL)
#define DWT_CTRL    (*(volatile uint32 *)0xE0001000UL)
#define DWT_CYCCNT  (*(volatile uint32 *)0xE0001004UL)
#define CORE_DEMCR  (*(volatile uint32 *)0xE000EDFCUL)
#define CORE_MHZ    120U


/* Tamaños de cada capa de la red */
#define N_IN   1U   /* neuronas de entrada                */
#define N_L1   2U   /* neuronas de la capa 1               */
#define N_L2   3U   /* neuronas de la capa 2               */
#define N_L3   1U   /* neuronas de la capa 3 (salida)      */

/* Definiciones de entradas y salidas para aplanar arreglo */
#define MAX_W    6U
#define MAX_OUT  3U

#define LEARNING_RATE 0.3f

/* Estructuras convenientes */
typedef struct {
    uint8   inputs;
    uint8   outputs;
    float32 weights[MAX_W];
    float32 bias[MAX_OUT];
} Layer;

typedef struct {
    float32 a;
    float32 b;
    float32 c;
    float32 d;
} zigmoid;

/* Cache del forward que guarda la activacion de la neurona
 * de cada capa, para poder calcular el backward despues.      */
typedef struct {
	float32 a1[N_L1];
	float32 a2[N_L2];
	float32 a3[N_L3];
} ForwardCache;


/*Funciones locales*/
void      initPeriphericals(void);
static void InitLayerFrom(Layer *L, uint8 inputs, uint8 outputs,
                          const float32 *w, const float32 *b);
float32   Neuron(zigmoid Zigmoid, float32 z);
static inline float32 NeuronDerFromA(zigmoid Z, float32 aOut);
void      Forward(const float32 *U,
                   const Layer *L1, const zigmoid *Act1,
                   const Layer *L2, const zigmoid *Act2,
                   const Layer *L3, const zigmoid *Act3,
                   ForwardCache *C);
void      Backward(const float32 *U,
                    Layer *L1, const zigmoid *Act1,
                    Layer *L2, const zigmoid *Act2,
                    Layer *L3, const zigmoid *Act3,
                    const ForwardCache *C,
                    float32 target, float32 lr);
float32 TrainStep(const float32 *U, float32 target, uint8 doUpdate,
                  Layer *L1, const zigmoid *Act1,
                  Layer *L2, const zigmoid *Act2,
                  Layer *L3, const zigmoid *Act3,
                  float32 lr, float32 *outPred,
                  uint32 *outCycFwd, uint32 *outCycBwd);
static uint8 Float2Str4(float32 value, char *buf, uint8 buf_size);
static uint8 U32ToStr(uint32 v, char *buf, uint8 buf_size);
static inline void DWT_Init(void);
static inline uint32 DWT_Cycles(void);

/*Globales*/
volatile uint8 u8BufferIdx;
volatile uint8 au8Buffer[BUFFER_SIZE]; //buffer de recepción que se construye en IRQ
volatile uint8 bRxFlag = 0U; // Flag de nueva recepción
static char   txBuf[48]; //buffer de envío
//Variables para obtención de ciclos forward y backward
static uint32 cyclesFwd = 0;
static uint32 cyclesBwd = 0;

/*Callback de interrupción*/
void Uart_Callback(const uint8 HwInstance, const Lpuart_Uart_Ip_EventType Event, const void *UserData)
    {
        switch (Event)
        {
            case LPUART_UART_IP_EVENT_RX_FULL:
                /* La recepción termina con salto de linea */
                if ((au8Buffer[u8BufferIdx] != '\n') && (u8BufferIdx != (BUFFER_SIZE - 2U)))
                {
                    /* Actuliza el buffer con el nuevo elemento en DATA */
                    u8BufferIdx++;
                    Lpuart_Uart_Ip_SetRxBuffer(UART_INSTANCE, (uint8 *)&au8Buffer[u8BufferIdx], 1U);
                }
                else
                {
                    /* Nueva recepción */
                    bRxFlag = 1;
                }
                break;

            case LPUART_UART_IP_EVENT_TX_EMPTY:
                __asm volatile ("nop");
                break;

            case LPUART_UART_IP_EVENT_END_TRANSFER:
                Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, (uint8 *)au8Buffer, 1U);
                break;

            case LPUART_UART_IP_EVENT_ERROR:
                __asm volatile ("nop");
                break;

            default:
                __asm volatile ("nop");
                break;
        }

        /* Evitar flags */
           (void)UserData;
           (void)HwInstance;
    }

int main(void)
{
    /* Inicializar perifericos */
    initPeriphericals();
    DWT_Init();

    /* Cada neurona tiene su propia forma de sigmoide parametrico (a,b,c,d).
     * a=1 en todas -> salida acotada a (0,1) en cada neurona,
     * b, c y d varian de neurona a neurona para que cada una sea
     * distinta tenido desface en ordenada y abcisa al origen.
     * Hay 2+3+1 = 6 neuronas en total cuyos parámetros son estaticos*/

    static const zigmoid Act1[N_L1] = { {1.0f, 1.0f, 0.5f, -1.0f},
                                        {1.0f, 1.0f, 0.5f,  1.0f} };
    static const zigmoid Act2[N_L2] = { {1.0f, 0.5f, 1.0f, 0.0f},
                                        {1.0f, 1.0f, 1.0f, 0.0f},
                                        {1.0f, 1.5f, 1.0f, 0.0f} };
    static const zigmoid Act3[N_L3] = { {1.0f, 1.0f, 1.5f, 0.0f} };

    /* Creamos las capas de la red con la estructura deseada  con pesos, y bias de inicio*/

    static Layer Layer1,Layer2,Layer3;    InitLayerFrom(&Layer1, N_IN, N_L1, W1_INIT, B1_INIT);
    InitLayerFrom(&Layer2, N_L1, N_L2, W2_INIT, B2_INIT);
    InitLayerFrom(&Layer3, N_L2, N_L3, W3_INIT, B3_INIT);

    /*Armamos la primer recepción*/
    Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, (uint8 *)au8Buffer, 1U);
    while (1)
    {
        if (bRxFlag == 1U)//Nueva recepción disponible
        {
        	bRxFlag = 0U;
            /* Protocolo de linea: "<modo>,<entrada>,<etiqueta>\n"
             *   modo 'T' -> entrena (forward + backward + actualiza pesos)
             *   modo 'V' -> solo evalua (forward-only, NO toca los pesos)
             * Ejemplo real: "T,-0.383,0\n" o "V,0.283,1\n"
             * (la entrada ya viene normalizada por el script de Python,
             * que aplica encode_ascii antes de transmitir).                */
            char    mode = 'T';
            float32 U[N_IN];
            float32 target;
            float32 pred;
            u8BufferIdx++;
            au8Buffer[u8BufferIdx] = 0U;
            int parsed = sscanf((const char *)au8Buffer, "%c,%f,%f", &mode, &U[0], &target);//Lemos el contenido y parseamos el buffer
            if (parsed != 3)//Validación de parsea de scan
            {
                u8BufferIdx = 0U;
                txBuf[0] = 'E'; txBuf[1] = '\n';
                (void)Lpuart_Uart_Ip_SyncSend(UART_INSTANCE, (uint8 *)txBuf,2U, 100000U);//enviar error
                continue;
            }

            uint8 doUpdate = ((mode == 'T') || (mode == 't')) ? 1U : 0U;//Booleano para determinar si pasa por backward

            //Aplicación de forward y backward para entrenamiento
            float32 loss = TrainStep(U, target, doUpdate,
                                      &Layer1, Act1,
                                      &Layer2, Act2,
                                      &Layer3, Act3,
                                      LEARNING_RATE, &pred,&cyclesFwd,&cyclesBwd);

            /* Responde "prediccion,perdida,ciclosF,ciclosB\n" para que la PC pueda calcular
             * accuracy (comparando prediccion contra la etiqueta) y graficar
             * la curva de perdida, sea en modo entrenamiento o validacion y obtener los
             * ciclos de procesamiento en ambas secuencias.  */

            uint8 n = 0U;
            n += Float2Str4(pred, &txBuf[n], (uint8)(sizeof(txBuf) - n));
            txBuf[n++] = ',';
            n += Float2Str4(loss, &txBuf[n], (uint8)(sizeof(txBuf) - n));
            txBuf[n++] = ',';
            n += U32ToStr(cyclesFwd, &txBuf[n], (uint8)(sizeof(txBuf) - n));
            txBuf[n++] = ',';
            n += U32ToStr(cyclesBwd, &txBuf[n], (uint8)(sizeof(txBuf) - n));
            txBuf[n++] = '\n';

            /* Timeout en microsegundos: ajusta segun tu baudrate real; con
             * ~20 bytes y 9600 baudios sobran 10-20 ms, aqui se deja
             * bastante margen. */
            (void)Lpuart_Uart_Ip_SyncSend(UART_INSTANCE, (uint8 *)txBuf,(uint32)n, 100000U);



            /* Reinicia la recepcion para la siguiente linea */
            u8BufferIdx = 0U;
            Lpuart_Uart_Ip_AsyncReceive(UART_INSTANCE, (uint8 *)au8Buffer, 1U);
        }
    }

    return 0;
}


void initPeriphericals(void){
    /* Inicialización de rutina de reloj */
    Clock_Ip_Init(Clock_Ip_aClockConfig);
    Clock_Ip_InitClock(Clock_Ip_aClockConfig);
    while(CLOCK_IP_PLL_LOCKED != Clock_Ip_GetPllStatus()){
        __asm volatile ("nop");
    }
    Clock_Ip_DistributePll();

    /* Inicializar interrupción */
    IntCtrl_Ip_Init(&UART_INTERRUPT);
    IntCtrl_Ip_EnableIrq(LPUART6_IRQn);

    /* Inicilizar GPIOS */
    Siul2_Port_Ip_Init(SIUL2_PINS, SIUL2_CONFIG);

    /* Inicializar UART */
    Lpuart_Uart_Ip_Init(UART_INSTANCE, &Lpuart_Uart_Ip_xHwConfigPB_6);
}

//inicilizar capas dadas las definiciones de struc layer y los pesos y bias de inicio
static void InitLayerFrom(Layer *L, uint8 inputs, uint8 outputs,
                          const float32 *w, const float32 *b)
{
    L->inputs  = inputs;
    L->outputs = outputs;
    for (uint8 k = 0U; k < (uint8)(inputs * outputs); k++) { L->weights[k] = w[k]; }
    for (uint8 j = 0U; j < outputs; j++)                   { L->bias[j]    = b[j]; }
}



/* Sigmoide parametrico: f(z) = a / (1 + b*e^(-c*z)) + d
 * acotado a 80.0f para evitar inf y por tanto NAN*/
float32 Neuron(zigmoid Zigmoid, float32 z)
{
    float32 arg = -Zigmoid.c * z;
    if (arg >  80.0f) { arg =  80.0f; }
    if (arg < -80.0f) { arg = -80.0f; }
    return (Zigmoid.a / (1.0f + (Zigmoid.b * expf(arg)))) + Zigmoid.d;
}

/* Derivada de Neurona postactivación, necesaria para el
 * backward: f'(z) = a*c*u * (1 - u), donde u = (activacion - d)/a
 * aplanado para evitar expf()                  */
static inline float32 NeuronDerFromA(zigmoid Z, float32 aOut)
{
    float32 u = (aOut - Z.d) / Z.a;
    return Z.a * Z.c * u * (1.0f - u);
}

/* Forward pass completo: calcula y a (postactivacion) de cada capa.
 * Usa Neuron (no NeuronDer) porque en el
 * forward lo que se propaga es la activacion, no su derivada.           */
void Forward(const float32 *U,
             const Layer *L1, const zigmoid *Act1,
             const Layer *L2, const zigmoid *Act2,
             const Layer *L3, const zigmoid *Act3,
             ForwardCache *C)
{
    for (uint8 j = 0U; j < L1->outputs; j++)
    {
        float32 z = L1->bias[j];
        for (uint8 i = 0U; i < L1->inputs; i++)
        {
            z += L1->weights[(j * L1->inputs) + i] * U[i];
        }
        C->a1[j] = Neuron(Act1[j], z);
    }

    for (uint8 j = 0U; j < L2->outputs; j++)
    {
        float32 z = L2->bias[j];
        for (uint8 i = 0U; i < L2->inputs; i++)
        {
            z += L2->weights[(j * L2->inputs) + i] * C->a1[i];
        }
        C->a2[j] = Neuron(Act2[j], z);
    }

    for (uint8 j = 0U; j < L3->outputs; j++)
    {
        float32 z = L3->bias[j];
        for (uint8 i = 0U; i < L3->inputs; i++)
        {
            z += L3->weights[(j * L3->inputs) + i] * C->a2[i];
        }
        C->a3[j] = Neuron(Act3[j], z);
    }
}

/* Backward pass (retropropagacion) con perdida MSE y descenso de
 * gradiente estocastico (un ejemplo a la vez).                          */
void Backward(const float32 *U,
              Layer *L1, const zigmoid *Act1,
              Layer *L2, const zigmoid *Act2,
              Layer *L3, const zigmoid *Act3,
              const ForwardCache *C,
              float32 target, float32 lr)
{
    float32 delta3[N_L3]={0.0f};
    float32 delta2[N_L2];
    float32 delta1[N_L1];

    /* --- Capa de salida: dLoss/dz3 = (a3 - target) * f'(z3) --- */
    for (uint8 j = 0U; j < L3->outputs; j++)
    {
        float32 error = C->a3[j] - target;
        delta3[j] = error * NeuronDerFromA(Act3[j],C->a3[j]);
    }

    /* --- Capa 2: se propaga el error hacia atras a traves de W3 --- */
    for (uint8 i = 0U; i < L2->outputs; i++)
    {
        float32 sum = 0.0f;
        for (uint8 j = 0U; j < L3->outputs; j++)
        {
            sum += L3->weights[(j * L3->inputs) + i] * delta3[j];
        }
        delta2[i] = sum * NeuronDerFromA(Act2[i],C->a2[i]);
    }

    /* --- Capa 1: se propaga el error hacia atras a traves de W2 --- */
    for (uint8 i = 0U; i < L1->outputs; i++)
    {
        float32 sum = 0.0f;
        for (uint8 j = 0U; j < L2->outputs; j++)
        {
            sum += L2->weights[(j * L2->inputs) + i] * delta2[j];
        }
        delta1[i] = sum * NeuronDerFromA(Act1[i],C->a1[i]);
    }

    /* --- Actualizacion de pesos y bias (descenso de gradiente) --- */
    for (uint8 j = 0U; j < L3->outputs; j++)
    {
        for (uint8 i = 0U; i < L3->inputs; i++)
        {
            L3->weights[(j * L3->inputs) + i] -= lr * delta3[j] * C->a2[i];
        }
        L3->bias[j] -= lr * delta3[j];
    }

    for (uint8 j = 0U; j < L2->outputs; j++)
    {
        for (uint8 i = 0U; i < L2->inputs; i++)
        {
            L2->weights[(j * L2->inputs) + i] -= lr * delta2[j] * C->a1[i];
        }
        L2->bias[j] -= lr * delta2[j];
    }

    for (uint8 j = 0U; j < L1->outputs; j++)
    {
        for (uint8 i = 0U; i < L1->inputs; i++)
        {
            L1->weights[(j * L1->inputs) + i] -= lr * delta1[j] * U[i];
        }
        L1->bias[j] -= lr * delta1[j];
    }
}

/* Une forward + (dado el byte de activación) backward + calculo de la perdida en un
 * solo paso. Con doUpdate=1 entrena (actualiza pesos); con doUpdate=0
 * solo evalua (util para medir accuracy en un set de validacion sin
 * seguir aprendiendo de esos datos).                                     */
float32 TrainStep(const float32 *U, float32 target, uint8 doUpdate,
                  Layer *L1, const zigmoid *Act1,
                  Layer *L2, const zigmoid *Act2,
                  Layer *L3, const zigmoid *Act3,
                  float32 lr, float32 *outPred,
                  uint32 *outCycFwd, uint32 *outCycBwd)
{
    ForwardCache C;

    uint32 c0 = DWT_Cycles();
    Forward(U, L1, Act1, L2, Act2, L3, Act3, &C);
    uint32 c1 = DWT_Cycles();

    if (doUpdate == 1U)
    {
        Backward(U, L1, Act1, L2, Act2, L3, Act3, &C, target, lr);
    }
    uint32 c2 = DWT_Cycles();

    *outCycFwd = c1 - c0;
    *outCycBwd = c2 - c1;

    *outPred = C.a3[0];
    float32 err = C.a3[0] - target;
    return 0.5f * err * err;
}

/* Parsing manual de flotantes actodaos para su envío por uart como caracter evita arrastre de printf*/
static uint8 Float2Str4(float32 value, char *buf, uint8 buf_size)
{
    uint8 pos = 0U;
    if (value < 0.0f) { if (pos < buf_size) { buf[pos++] = '-'; } value = -value; }

    uint32 scaled   = (uint32)((value * 10000.0f) + 0.5f);   /* 4 decimales */
    uint32 intPart  = scaled / 10000U;
    uint32 fracPart = scaled % 10000U;

    char  tmp[12];
    uint8 n = 0U;
    if (intPart == 0U) { tmp[n++] = '0'; }
    while (intPart > 0U) { tmp[n++] = (char)('0' + (intPart % 10U)); intPart /= 10U; }
    while ((n > 0U) && (pos < buf_size)) { buf[pos++] = tmp[--n]; }

    if (pos < buf_size) { buf[pos++] = '.'; }
    for (sint32 div = 1000; div >= 1; div /= 10)
    {
        if (pos < buf_size) { buf[pos++] = (char)('0' + ((fracPart / (uint32)div) % 10U)); }
    }
    return pos;
}

/* Parsing manual de enteros actodaos para su envío por uart como caracter*/
static uint8 U32ToStr(uint32 v, char *buf, uint8 buf_size)
{
    char  tmp[12];
    uint8 n = 0U, pos = 0U;
    if (v == 0U) { tmp[n++] = '0'; }
    while (v > 0U) { tmp[n++] = (char)('0' + (v % 10U)); v /= 10U; }
    while ((n > 0U) && (pos < buf_size)) { buf[pos++] = tmp[--n]; }
    return pos;
}
//Iniciliza DWT para tiempos de ciclo
static inline void DWT_Init(void)
{
    CORE_DEMCR |= (1UL << 24);      /* TRCENA */
    DWT_LAR     = 0xC5ACCE55UL;     /* desbloqueo (requerido en algunos M7) */
    DWT_CYCCNT  = 0U;
    DWT_CTRL   |= 1UL;              /* CYCCNTENA */
}
//Retorna el valor de tiempos de ciclo
static inline uint32 DWT_Cycles(void) { return DWT_CYCCNT; }

#ifdef __cplusplus
}
#endif

/** @} */
