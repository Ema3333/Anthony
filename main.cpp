#include <iostream>
#include <fstream>
#include <cmath>
#include <numbers>
#include <random>

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
    float* weight1;
    float* bias1;
    float* weight2;
    float* bias2;
};

struct layerNorm
{
    float* gamma;
    float* beta;
};

struct trainData {
    // Stati del fiume residual
    float* x_in;
    float* x_norm1;
    float* x_after_attn;
    float* x_norm2;
    float* x_after_ffn;

    // Statistiche di LayerNorm
    float* mean_norm1;
    float* var_norm1;
    float* mean_norm2;
    float* var_norm2;

    // Attivazioni interne dell'attention
    float* Q;
    float* K;
    float* V;
    float* attn_weights;
    float* attn_output;

    // Attivazioni interne dell'FFN
    float* ffn_pre_gelu;
    float* ffn_post_gelu;
};

//declare global variables

config Config; // declare configuration variable with all the basic data to run the model

//declare functions prototypes
float* normalizer(float* transfBuild, layerNorm layerData, trainData train);
void declareEmbed(float* tokenEmbed, float* posEmbed, normal_distribution<float>, mt19937);
layerAttention declareAttention(layerAttention Layer, normal_distribution<float>, mt19937);
layerNorm declareNorm(layerNorm Layer);
layerFfn declareFfn(layerFfn LayerFfn, normal_distribution<float>, mt19937);
float* transformerNetwork(int tokenArr[1024], float* tokenEmbed, float* posEmbed);
float* attentionNetwork(layerAttention Layer, float* matTransf, float* residual, trainData train);
float* ffnNetwork(float* transf, layerFfn Layer, float* residual, trainData train);
int prediction(float* context, float* Layer);
float* transformerBlock(float* transfBuild, layerAttention layerAtt, layerFfn LayerFfn, layerNorm LayerNormAtt, layerNorm LayerNormFfn, trainData& train);

int main()
{
    mt19937 gen = mt19937(32);
    normal_distribution<float> gauss = normal_distribution<float>(0.0f, 0.02f);

    float* tokenEmbed = new float[config::vocab_size * config::d_model];
    float* posEmbed = new float[config::max_tok * config::d_model];
    float* transf = new float[config::seq_len * config::d_model];

    layerAttention layerAtt[config::n_layer];
    layerFfn LayerFfn[config::n_layer];
    layerNorm LayerNormAtt[config::n_layer];
    layerNorm LayerNormFfn[config::n_layer];
    layerNorm layerNormFinal;
    trainData* train = new trainData[config::n_layer];

    declareEmbed(tokenEmbed, posEmbed, gauss, gen);
    transf = transformerNetwork(tokenArr, tokenEmbed, posEmbed);

    for (int i = 0; i < config::n_layer; i++)
    {
        layerAtt[i] = declareAttention(layerAtt[i], gauss, gen);
        LayerFfn[i] = declareFfn(LayerFfn[i], gauss, gen);
        LayerNormAtt[i] = declareNorm(LayerNormAtt[i]);
        LayerNormFfn[i] = declareNorm(LayerNormFfn[i]);
    }

    for (int i = 0; i < config::n_layer; i++)
    {
        transf = transformerBlock(transf, layerAtt[i], LayerFfn[i], LayerNormAtt[i], LayerNormFfn[i], train[i]);
    }
    delete[] posEmbed;

    layerNormFinal = declareNorm(layerNormFinal);
    float* transfNorm = normalizer(transf, layerNormFinal, train[config::n_layer - 1]);

    prediction(transfNorm, tokenEmbed);
}

float* normalizer(float* transfBuild, layerNorm layerData, trainData train)
{
    float mean = 0;
    float variance = 0;
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
            normTransfOut[a * config::d_model + i] = (transfBuild[a * config::d_model + i] - mean) / sqrt(variance + 1e-5);
            normTransfOut[a * config::d_model + i] = layerData.gamma[i] * normTransfOut[a * config::d_model + i] + layerData.beta[i];
        }

        variance = 0;
        mean = 0;
    }

    return normTransfOut;
}

void declareEmbed(float* tokenEmbed, float* posEmbed, normal_distribution<float> gauss, mt19937 gen)
{
    for (int i = 0; i < config::vocab_size; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            tokenEmbed[i * config::d_model + j] = gauss(gen);
        }
    }

    for (int i = 0; i < config::max_tok; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            posEmbed[i * config::d_model + j] = gauss(gen);
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

layerAttention declareAttention(layerAttention Layer, normal_distribution<float> gauss, mt19937 gen)
{
    Layer.Wq = new float[config::d_model * config::d_model];
    Layer.Wk = new float[config::d_model * config::d_model];
    Layer.Wv = new float[config::d_model * config::d_model];
    Layer.Wo = new float[config::d_model * config::d_model];

    for (int i = 0; i < config::d_model; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            Layer.Wq[i * config::d_model + j] = gauss(gen);
            Layer.Wk[i * config::d_model + j] = gauss(gen);
            Layer.Wv[i * config::d_model + j] = gauss(gen);
            Layer.Wo[i * config::d_model + j] = gauss(gen);
        }
    }

    return Layer;
}

layerFfn declareFfn(layerFfn LayerFfn, normal_distribution<float> gauss, mt19937 gen)
{
    LayerFfn.weight1 = new float[config::d_model * config::d_model * 4];
    LayerFfn.bias1 = new float[config::d_model * 4];
    LayerFfn.weight2 = new float[config::d_model * config::d_model * 4];
    LayerFfn.bias2 = new float[config::d_model];

    for (int i = 0; i < config::d_model; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            LayerFfn.weight1[i * (config::d_model * 4) + j] = gauss(gen);
            if (i == 0)
            {
                LayerFfn.bias1[j] = 0.0;
            }
        }
    }

    for (int i = 0; i < config::d_model * 4; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            LayerFfn.weight2[i * config::d_model + j] = gauss(gen);
            if (i == 0)
            {
                LayerFfn.bias2[j] = 0.0;
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
        for (int j = 0; j < config::d_model; j++)
        {
            transformerIn[i * config::d_model + j] = tokenEmbed[tokenArr[i] * config::d_model + j] + posEmbed[i * config::d_model + j];
        }
    }

    return transformerIn;
}

float* attentionNetwork(layerAttention Layer, float* matTransf, float* residual, trainData train)
{
    float* attentionQ = new float[config::seq_len * config::d_model];
    float* attentionK = new float[config::seq_len * config::d_model];
    float* attentionV = new float[config::seq_len * config::d_model];
    float* score = new float[config::seq_len * config::seq_len];
    float* output = new float[config::seq_len * config::d_model];
    float* outputProjected = new float[config::seq_len * config::d_model];
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
                temp += matTransf[i * config::d_model + k] * Layer.Wk[k * config::d_model + j];
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

    float* max = new float[config::seq_len];
    float* sum = new float[config::seq_len];

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
            outputProjected[i * config::d_model + j] += residual[i * config::d_model + j];
        }
    }

    delete[] attentionQ;
    delete[] attentionK;
    delete[] attentionV;
    delete[] score;
    delete[] output;
    delete[] max;
    delete[] sum;

    return outputProjected;
}

float* ffnNetwork(float* transf, layerFfn Layer, float* residual, trainData train)
{
    float* expTansf = new float[config::seq_len * config::d_model * 4];
    float* resizeTransf = new float[config::seq_len * config::d_model];

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            expTansf[i * (config::d_model * 4) + j] = 0.0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            resizeTransf[i * config::d_model + j] = 0.0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                expTansf[i * (config::d_model * 4) + j] += transf[i * config::d_model + k] * Layer.weight1[k * (config::d_model * 4) + j];
            }

            expTansf[i * (config::d_model * 4) + j] += Layer.bias1[j];
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model * 4; j++)
        {
            float x = expTansf[i * (config::d_model * 4) + j];
            expTansf[i * (config::d_model * 4) + j] = x * 0.5 * (1 + tanh(sqrt(2/numbers::pi) * (x + 0.044715 * x * x * x)));
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            for (int k = 0; k < config::d_model * 4; k++)
            {
                resizeTransf[i * config::d_model + j] += expTansf[i * (config::d_model * 4) + k] * Layer.weight2[k * config::d_model + j];
            }

            resizeTransf[i * config::d_model + j] += Layer.bias2[j];
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::d_model; j++)
        {
            resizeTransf[i * config::d_model + j] += residual[i * config::d_model + j];
        }
    }

    delete[] expTansf;

    return resizeTransf;
}

int prediction(float* transf, float* tokenEmbed)
{
    float* logits = new float[config::seq_len * config::vocab_size];
    int predictedToken = 0;

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::vocab_size; j++)
        {
            logits[i * config::vocab_size + j] = 0.0;
        }
    }

    for (int i = 0; i < config::seq_len; i++)
    {
        for (int j = 0; j < config::vocab_size; j++)
        {
            for (int k = 0; k < config::d_model; k++)
            {
                logits[i * config::vocab_size + j] += transf[i * config::d_model + k] * tokenEmbed[j * config::d_model + k];
            }
        }
    }

    float max;
    float sum;
    for (int i = 0; i < config::seq_len; i++)
    {
        sum = 0.0;
        max = logits[i * config::vocab_size + 0];

        for (int j = 0; j < config::vocab_size; j++)
        {
            if (logits[i * config::vocab_size + j] > max)
            {
                max = logits[i * config::vocab_size + j];
            }
        }

        for (int j = 0; j < config::vocab_size; j++)
        {
            logits[i * config::vocab_size + j] = exp(logits[i * config::vocab_size + j] - max);
            sum += logits[i * config::vocab_size + j];
        }

        for (int j = 0; j < config::vocab_size; j++)
        {
            logits[i * config::vocab_size + j] /= sum;
        }
    }

    max = 0.0;
    for (int j = 0; j < config::vocab_size; j++)
    {
        if (max < logits[(config::seq_len - 1) * config::vocab_size + j])
        {
            max = logits[(config::seq_len - 1) * config::vocab_size + j];
            predictedToken = j;
        }
    }

    delete[] logits;

    cout << predictedToken << ", " << max << endl;

    return predictedToken;
}

float* transformerBlock(float* transfBuild, layerAttention layerAtt, layerFfn LayerFfn, layerNorm LayerNormAtt, layerNorm LayerNormFfn, trainData& train)
{
    train.x_in = transfBuild;

    float* norm1 = normalizer(transfBuild, LayerNormAtt, train);
    train.x_norm1 = norm1;

    float* resultAtt = attentionNetwork(layerAtt, norm1, transfBuild, train);
    train.x_after_attn = resultAtt;

    float* norm2 = normalizer(resultAtt, LayerNormFfn, train);
    train.x_norm2 = norm2;

    float* resultFfn = ffnNetwork(norm2, LayerFfn, resultAtt, train);
    train.x_after_ffn = resultFfn;

    return resultFfn;
}

void emptyTrainData(trainData* train, int n_layer)
{
    for (int b = 0; b < n_layer; b++)
    {
        delete[] train[b].x_norm1;
        delete[] train[b].x_after_attn;
        delete[] train[b].x_norm2;
        delete[] train[b].x_after_ffn;

        delete[] train[b].mean_norm1;
        delete[] train[b].var_norm1;
        delete[] train[b].mean_norm2;
        delete[] train[b].var_norm2;

        delete[] train[b].Q;
        delete[] train[b].K;
        delete[] train[b].V;
        delete[] train[b].attn_weights;
        delete[] train[b].attn_output;

        delete[] train[b].ffn_pre_gelu;
        delete[] train[b].ffn_post_gelu;
    }
}
