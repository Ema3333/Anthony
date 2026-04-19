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

struct layerAttention
{
    float* Wq;
    float* Wk;
    float* Wv;
    float* Wo;
};

struct layerFfn
{
    float* weight;
    float* bias;
};

struct layerNorm
{
    float* gamma;
    float* beta;
};

//declare global variables

config Config; // declare configuration variable with all the basic data to run the model

//declare functions prototypes
float* normalizer(float* transfBuild, layerNorm layerData);
void declareEmbed(float* tokenEmbed, float* posEmbed);
layerAttention declareAttention(layerAttention Layer);
layerNorm declareNorm(layerNorm Layer);
layerFfn declareFfn(layerFfn LayerFfn);
float* transformerNetwork(int tokenArr[1024], float* tokenEmbed, float* posEmbed);
float* attentionNetwork(layerAttention Layer, float* matTransf);
float* ffnNetwork(float* data, layerFfn Layer);

int main()
{
    srand(time(NULL));

    float* tokenEmbed = new float[config::vocab_size * config::d_model];
    float* posEmbed = new float[config::max_tok * config::d_model];

    declareEmbed(tokenEmbed, posEmbed);
    layerAttention layerAtt;
    layerFfn LayerFfn;
    layerNorm LayerNormAtt;
    layerNorm LayerNormFfn;
    layerAtt = declareAttention(layerAtt);
    LayerNormAtt = declareNorm(LayerNormAtt);

    float* transfBuild = transformerNetwork(tokenArr, tokenEmbed, posEmbed);
    transfBuild = normalizer(transfBuild, LayerNormAtt);
    float* result = attentionNetwork(layerAtt, transfBuild);

    LayerNormFfn = declareNorm(LayerNormFfn);
    float* resultNorm = normalizer(result, LayerNormFfn);
    LayerFfn = declareFfn(LayerFfn);
    ffnNetwork(resultNorm, LayerFfn);
}

float* normalizer(float* transfBuild, layerNorm layerData)
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

void declareEmbed(float* tokenEmbed, float* posEmbed)
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



layerNorm declareNorm(layerNorm Layer)
{
    Layer.gamma = new float[config::d_model];
    Layer.beta = new float[config::d_model];

    for (int i = 0; i < config::d_model; i++)
    {
        Layer.gamma[i] = 1;
        Layer.beta[i] = 0;
    }

    return Layer;
}

layerAttention declareAttention(layerAttention Layer)
{
    Layer.Wq = new float[config::d_model * config::d_model];
    Layer.Wk = new float[config::d_model * config::d_model];
    Layer.Wv = new float[config::d_model * config::d_model];
    Layer.Wo = new float[config::d_model * config::d_model];

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

layerFfn declareFfn(layerFfn LayerFfn)
{
    LayerFfn.weight = new float[config::d_model * config::d_model * 4];
    LayerFfn.bias = new float[config::d_model * 4];

    for (int i = 0; i < config::d_model; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            LayerFfn.weight[i * (config::d_model * 4) + j] = rand() % 200 / 100.0;
            if (i == 0)
            {
                LayerFfn.bias[j] = 0.0;
            }
        }
    }

    return LayerFfn;
}

float* transformerNetwork(int tokenArr[1024], float* tokenEmbed, float* posEmbed)
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

float* attentionNetwork(layerAttention Layer, float* matTransf)
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

    return result;
}

float* ffnNetwork(float* transf, layerFfn Layer)
{
    float* expTansf = new float[config::seq_len * config::d_model * 4];

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            expTansf[i * (config::d_model * 4) + j] = 0.0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                expTansf[i * (config::d_model * 4) + j] += transf[i * config::d_model + k] * Layer.weight[k * (config::d_model * 4) + j];
            }

            expTansf[i * (config::d_model * 4) + j] += Layer.bias[j];
        }
    }
}
