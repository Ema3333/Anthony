#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>

using namespace std;

int tokenArr[] = {40, 1842, 616, 3290};

//declare struct
struct config
{
    static const int vocab_size = 50257;
    static const int d_model = 256;
    static const int n_layer = 12;
    static const int n_head = 12;
    static const int max_tok = 1024;
    static const int seq_len = sizeof(tokenArr) / sizeof(tokenArr[0]);
};

struct layer
{
    float* Wq;
    float* Wk;
    float* Wv;
    float* Wo;
    float* gamma;
    float* beta;
};

//declare global variables

config Config; // declare configuration variable with all the basic data to run the model

//declare functions prototypes
void rndDeclare(float* tokenEmbed, float* posEmbed);
layer declareLayer(layer Layer);
float* buildTransformerIn(int tokenArr[1024], float* tokenEmbed, float* posEmbed);
float* normalizeTransformer(float* transfBuild, layer layerData);

int main()
{
    srand(time(NULL));

    float* tokenEmbed = new float[config::vocab_size * config::d_model];
    float* posEmbed = new float[config::max_tok * config::d_model];

    rndDeclare(tokenEmbed, posEmbed);
    layer Layer;
    Layer = declareLayer(Layer);

    float* transfBuild = buildTransformerIn(tokenArr, tokenEmbed, posEmbed);
    float* addAttention(transfBuild);

    normalizeTransformer(transfBuild, Layer);
}

void rndDeclare(float* tokenEmbed, float* posEmbed)
{
    for (int i = 0; i < config::vocab_size; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            tokenEmbed[i * config::d_model + j] = rand() % 200 / 100.0;
        }
    }

    for (int i = 0; i < config::max_tok; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            posEmbed[i * config::d_model + j] = rand() % 200 / 100.0;
        }
    }
}

layer declareLayer(layer Layer)
{
    Layer.gamma = new float[config::d_model];
    Layer.beta = new float[config::d_model];

    Layer.Wq = new float[config::d_model * config::d_model];
    Layer.Wk = new float[config::d_model * config::d_model];
    Layer.Wv = new float[config::d_model * config::d_model];
    Layer.Wo = new float[config::d_model * config::d_model];

    for (int i = 0; i < config::d_model; i++)
    {
        Layer.gamma[i] = 1;
        Layer.beta[i] = 0;
    }

    for (int i = 0; i < config::d_model; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            Layer.Wq[i * config::d_model + j] = rand() % 200 / 100.0 - 100.0;
            Layer.Wk[i * config::d_model + j] = rand() % 200 / 100.0 - 100.0;
            Layer.Wv[i * config::d_model + j] = rand() % 200 / 100.0 - 100.0;
            Layer.Wo[i * config::d_model + j] = rand() % 200 / 100.0 - 100.0;
        }
    }

    return Layer;
}

float* buildTransformerIn(int tokenArr[1024], float* tokenEmbed, float* posEmbed)
{
    float* transformerIn = new float[config::seq_len * config::d_model];

    for (int i = 0; i < config::seq_len; i++)
    {
        cout << endl;
        for (int j = 0; j < config::d_model; j++)
        {
            transformerIn[i * config::d_model + j] = tokenEmbed[tokenArr[i] * config::d_model + j] + posEmbed[i * config::d_model + j];
        }
    }

    return transformerIn;
}

float* normalizeTransformer(float* transfBuild, layer layerData)
{
    float mean = 0;
    float variance = 0;
    float* normTransf = new float[config::seq_len * config::d_model];
    float* normTransfOut = new float[config::seq_len * config::d_model];

    for (int a = 0; a < config::seq_len; a++)
    {
        for (int i = 0; i < config::d_model; i++)
        {
            mean += transfBuild[a * config::d_model + i];
        }

        mean /= config::d_model;
        for (int i = 0; i < config::d_model; i++)
        {
            variance += pow(transfBuild[a * config::d_model + i] - mean, 2);
        }

        variance /= config::d_model;
        for (int i = 0; i < config::d_model; i++)
        {
            normTransf[a * config::d_model + i] = (transfBuild[a * config::d_model + i] - mean) / sqrt(variance + 1e-5);
            normTransfOut[a * config::d_model + i] = layerData.gamma[i] * normTransf[a * config::d_model + i] + layerData.beta[i];
        }

        variance = 0;
        mean = 0;
    }

    return normTransfOut;
}

float* addAttention(layer Layer, float* matTransf)
{
    float* attentionQ = new float[config::seq_len * config::d_model];
    float* attentionK = new float[config::seq_len * config::d_model];
    float* attentionV = new float[config::seq_len * config::d_model];
    float* score = new float[config::seq_len * config::seq_len];
    float* output = new float[config::seq_len * config::d_model];
    float* outputProjected = new float[config::seq_len * config::d_model];
    float* result = new float[config::seq_len * config::d_model];
    float temp = 0;

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                temp += matTransf[i * config::d_model + k] * Layer.Wq[k * config::d_model + j];
            }

            attentionQ[i * config::d_model + j] = temp;
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                temp += attentionK[i * config::d_model + k] * attentionQ[j * config::d_model + k];
            }

            attentionK[i * config::d_model + j] = temp;
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                temp += matTransf[i * config::d_model + k] * Layer.Wv[k * config::d_model + j];
            }

            attentionV[i * config::d_model + j] = temp;
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::seq_len; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                temp += attentionQ[i * config::d_model + k] * attentionK[j * config::d_model + k];
            }
            score[i * config::seq_len + j] = temp / sqrt(config::d_model);
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::seq_len; j++)
        {
            if (i < j)
            {
                score[i * config::seq_len + j] = -1e9;
            }
        }
    }

    float max[config::seq_len];
    float sum[config::seq_len];

    for (int i = 0; i < config::seq_len; i++)
    {
        max[i] = -1e9;
        sum[i] = 0;
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::seq_len; j++)
        {
            if (score[i * config::seq_len + j] > max[i])
            {
                max[i] = score[i * config::seq_len + j];
            }
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::seq_len; j++)
        {
            score[i * config::seq_len + j] = exp(score[i * config::seq_len + j] - max[i]);
            sum[i] += score[i * config::seq_len + j];
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::seq_len; j++)
        {
            score[i * config::seq_len + j] /= sum[i];
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::seq_len; k++)
            {
                temp += score[i * config::seq_len + k] * attentionV[k * config::d_model + j];
            }

            output[i * config::d_model + j] = temp;
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                temp += output[i * config::d_model + k] * Layer.Wo[k * config::d_model + j];
            }

            outputProjected[i * config::d_model + j] = temp;
            temp = 0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            result[i * config::d_model + j] = matTransf[i * config::d_model + j] + outputProjected[i * config::d_model + j];
        }
    }
}
