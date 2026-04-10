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
    float* weight;
    float* bias;
    float* gamma;
    float* beta;
};

//declare global variables

config Config; // declare configuration variable with all the basic data to run the model
layer Layer;

//declare functions prototypes
void rndDeclare(float* tokenEmbed, float* posEmbed);
float* buildTransformerIn(int tokenArr[1024], float* tokenEmbed, float* posEmbed);
float* normalizeTransformer(float* transfBuild);

int main()
{
    srand(time(NULL));

    Layer.gamma = new float[config::seq_len * config::d_model];
    Layer.beta = 0;

    float* tokenEmbed = new float[config::vocab_size * config::d_model];
    float* posEmbed = new float[config::max_tok * config::d_model];

    rndDeclare(tokenEmbed, posEmbed);
    float* transfBuild = buildTransformerIn(tokenArr, tokenEmbed, posEmbed);

    normalizeTransformer(transfBuild);
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