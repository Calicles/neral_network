#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

using namespace std;

class Neuron;
class NNet;
class LCG;

struct Connection
{
    double weight;
    double delta_weight;
};

typedef vector<Neuron> Layer;

//++++++++++++++    class LCG    +++++++++++++++
class LCG
{
public:
    double getWeight(void);

private:
    unsigned seed = 1103527590;
    unsigned mod = pow(2, 31);
    unsigned coef = 1103515245;
    unsigned add = 12345;
    double norm = 0x7fffffff;
    double nextStep(void);
};

double LCG::nextStep(void)
{
    seed = (coef * seed + add) % mod;
}

double LCG::getWeight(void)
{
    double d = seed;
    nextStep();
    
    d /= norm;
    return d;
}

 
//++++++++++++++    class Neuron    +++++++++++++++
class Neuron 
{
public:
    Neuron(unsigned myIndex);
    void setOutputVal(double val) {m_outputVal = val;}
    void setConnection(double weightVal);
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer &prevLayer);
    void calcOutputGradient(const double targetVal);
    void calcHiddenGradient(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
    
private:
    static double eta;
    static double transfertFunction(double x);
    static double transfertFunctionDerivative(double x);
    static double randomWeight(void); //TODO use LCG
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    unsigned my_index;
    double m_gradient;
    vector<Connection> m_outputWeights;
};

double Neuron::eta = 0.5;

Neuron::Neuron(unsigned myIndex)
{
    my_index = myIndex;
}

void Neuron::setConnection(double weightVal)
{
    m_outputWeights.push_back(Connection());

    m_outputWeights.back().weight = weightVal;
}

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    
    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[my_index].weight;
    }
    
    m_outputVal = Neuron::transfertFunction( sum );
}

void Neuron::calcOutputGradient(const double targetVal)
{
    m_gradient = m_outputVal * (1 - m_outputVal) * (m_outputVal - targetVal);
}

void Neuron::calcHiddenGradient(const Layer &nextLayer)
{
    m_gradient = m_outputVal * (1 - m_outputVal) * sumDOW(nextLayer);
}

void Neuron::updateInputWeights(Layer &prevLayer) //TODO replace with game function
{
    // The weights to be updated are in the Connection container
    // in the neurons in the preceding layer
    for (unsigned n = 0; n < prevLayer.size() ; ++n) {
        Neuron &neuron = prevLayer[n];
        
        double newDeltaWeight = 
            // Individual input, magnified by the gradient and train rate:
            -eta 
            * neuron.getOutputVal()
            * m_gradient;
        
        neuron.m_outputWeights[my_index].delta_weight = newDeltaWeight;
        neuron.m_outputWeights[my_index].weight += newDeltaWeight;
        /*cerr << "neuron: " << n << endl;
        cerr << "weight: " << neuron.m_outputWeights[my_index].weight << endl;
        cerr << "outputval: " << neuron.getOutputVal() << endl;*/
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    // Sum our contributions of the errors at the nodes we feed.
    
    for (unsigned n = 0; n < nextLayer.size(); ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    
    return sum;
}

double Neuron::randomWeight(void) {
    return 0.0; //TODO implement LCG
}

double Neuron::transfertFunction(double x)
{
    return 1 / (1 + exp(-x));
}

double Neuron::transfertFunctionDerivative(double x)
{
    return Neuron::transfertFunction(1 - Neuron::transfertFunction(x));
}


//++++++++++++++    class NNet    +++++++++++++++
class NNet 
{
public:
    NNet(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) {return m_recentAverageError;}

private:
    LCG lcg;
    vector<Layer> m_layers; //layers[layers_num][node_num]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;
};

double NNet::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

NNet::NNet(const vector<unsigned> &topology)
{
    unsigned num_layers = topology.size();
    
    //Create and fill the layers with neurons, except the output layer
    for (unsigned layer_num = 0; layer_num < num_layers - 1; ++layer_num) {
        m_layers.push_back(Layer());
        
        // We have a new layer, now fill it with neurons, and
        // add a bias neuron in each layer except the last layer.
        for (unsigned neuron_num = 0; neuron_num <= topology[layer_num]; ++neuron_num) {
            m_layers.back().push_back(Neuron(neuron_num));
            //cerr << "Neuron made!" <<endl;
        }
        
        m_layers.back().back().setOutputVal(1.0);
    }
    
    // Fill the output layer without bias
    m_layers.push_back(Layer());
    for (unsigned neuron_num = 0; neuron_num < topology[num_layers - 1]; ++neuron_num) {
        m_layers.back().push_back(Neuron(neuron_num));
        //cerr << "Neuron made!" << endl;
    }
    
    
    //Fill the neuron's weight with the LCG
    for (unsigned layer_num = 0; layer_num < num_layers - 1; ++layer_num) {
        
        unsigned num_outputs = layer_num == num_layers - 1 ? 0 : m_layers[layer_num + 1].size();
        
        for (unsigned o = 0; o < num_outputs; ++o) {
            
            for (unsigned n = 0; n < m_layers[layer_num].size(); ++n) {
                double weight = lcg.getWeight();
                m_layers[layer_num][n].setConnection(weight);
                //cerr << "neuron: " << n << " in layer: " << layer_num << " fill with: " << weight << endl;
            }
        }
    }
}

void NNet::feedForward(const vector<double> &inputVals)
{
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }
    
    //Forward propagate
    for (unsigned layer_num = 1; layer_num < m_layers.size(); ++layer_num) {
        Layer &prevLayer = m_layers[layer_num - 1];
        
        for (unsigned n = 0; n < m_layers[layer_num].size(); ++n) {
            m_layers[layer_num][n].feedForward(prevLayer);
        }
    }
}

void NNet::backProp(const vector<double> &targetVals) //TODO use the game algo
{
    // Calculate overall net error (RMS of output neuron errors)
    
    Layer &outputLayer = m_layers.back();
    
    m_error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        
        m_error += delta * delta;
    }
    
    m_error /= outputLayer.size() - 1; //get average squared
    m_error = sqrt(m_error); //RMS
    
    m_recentAverageError = 
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) 
        / (m_recentAverageSmoothingFactor + 1.0);
        
    // Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size(); ++n) {
        outputLayer[n].calcOutputGradient(targetVals[n]);
    }
    
    // Calculate hidden layer gradients
    for (unsigned layer_num = m_layers.size() - 2; layer_num > 0; --layer_num) {
        Layer &hiddenLayer = m_layers[layer_num];
        Layer &nextLayer = m_layers[layer_num + 1];
        
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradient(nextLayer);
        }
    }
    
    // For all layers from outputs to first hidden layer,
    // update connection weights
    for (unsigned layer_num = m_layers.size() - 1; layer_num > 0; --layer_num) {
        Layer &layer = m_layers[layer_num];
        Layer &prevLayer = m_layers[layer_num - 1];
        
        //cerr << "layer: " << (layer_num-1) << endl;
        for (unsigned n = 0; n < layer.size(); ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NNet::getResults(vector<double> &resultVals) const
{
    resultVals.clear();
    
    for (unsigned n = 0; n < m_layers.back().size(); ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void showVectorVals(const string &label, const vector<double> &val)
{
    cerr << label;
    for (int i = 0; i < val.size(); ++i){
        cerr << val[i]  << " ";
    }
    cerr << endl;
}

void out(const vector<double> &val)
{
    for(int i = 0; i < val.size(); ++i){
        //int res = round(val[i]);
        int res = (val[i] < 0.49)? 0 : 1;
        cout << res;
    }
    cout << endl;
}


int main()
{
    int inputs;
    int outputs;
    int hiddenLayers;
    int testInputs;
    int trainingExamples;
    int trainingIterations;
    vector<unsigned> topology;
    cin >> inputs >> outputs >> hiddenLayers >> testInputs >> trainingExamples >> trainingIterations; cin.ignore();
    topology.push_back(inputs);
    
    cerr << "train ex : " << trainingExamples << " iteration: " << trainingIterations << endl;
    
    for (int i = 0; i < hiddenLayers; i++) {
        int nodes;
        cin >> nodes; cin.ignore();
        
        topology.push_back(nodes);
    }
    
    topology.push_back(outputs);
    
    cerr << "topology: ";
    for (unsigned i = 0; i < topology.size(); ++i) cerr << topology[i] << " ";
    cerr << endl;
    
    NNet net(topology);
    
    vector<double> testData;
    vector<double> trainInputs;
    vector<double> results;
    vector<double> targets;
    
    //Store test inputs
    cerr << "testInputs: " << testInputs << endl;
    for (int i = 0; i < testInputs; i++) {
        string testInput;
        //getline(cin, testInput);
        cin >> testInput; cin.ignore();
        cerr << "test: " << testInput << endl;
        double oneValue;
        for (int j = 0; j < testInput.length(); ++j) {
            oneValue = double(testInput[j] - 48);
            testData.push_back(oneValue);
        }
    }
    
    //Store the training data
    for (int i = 0; i < trainingExamples ; ++i) {
        string trainingInputs;
        string expectedOutputs;
        
        cin >> trainingInputs >> expectedOutputs; cin.ignore();
        
        double oneValue;
        cerr << "storing insputs: ";
        for (unsigned j = 0; j < inputs; ++j)
        {
            char c = trainingInputs[j];
            oneValue = double(c - 48);
            cerr << oneValue << " ";
            trainInputs.push_back(oneValue);
        }
        cerr << endl;
        cerr << "storing output: ";
        for (unsigned j = 0; j < outputs; ++j)
        {
            char c = expectedOutputs[j];
            oneValue = double(c - 48);
            cerr << oneValue << " ";
            targets.push_back(oneValue);
        }
        cerr << endl;
    }
    
    //Start training
    for (int i = 0; i < trainingIterations +2000; ++i) {
        vector<double> in;
        vector<double> tar;
        
        for (unsigned j = 0; j < trainingExamples; ++j)
        {
        
            unsigned start = j * inputs;
            unsigned end = start + inputs;
            for (unsigned k = start; k < end; ++k)
            {
                double oneValue = trainInputs[k];
                in.push_back(oneValue);
            }
            
            start = j * outputs;
            end = start + outputs;
            for (unsigned k = start; k < end; ++k)
            {
                double oneValue = targets[k];
                tar.push_back(oneValue);
            }
            net.feedForward(in);
            net.getResults(results);
            
            net.backProp(tar);
            
            if (i == trainingIterations - 1 + 2000){
            showVectorVals("inputs: ", in);
            showVectorVals("results: ", results);
            showVectorVals("target: ", tar);
            }
            
            in.clear();
            tar.clear();
            results.clear();
        }
    }
    
    //Test
    for (int i = 0; i < testInputs; i++) {
        vector<double> test;
        unsigned start = i * inputs;
        
        for (unsigned j = 0; j < inputs; ++j)
        {
            double oneValue = testData[start + j];
            test.push_back(oneValue);
        }
        
        net.feedForward(test);
        net.getResults(results);
        showVectorVals("test: ", test);
        out(results);

        results.clear();
    }
}
