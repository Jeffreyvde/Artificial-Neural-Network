using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    class Program
    {
        static void Main()
        {
            Sigmoid value = new Sigmoid();
            NeuralNetwork netowrk = new NeuralNetwork(new InputLayer(1), new OutputLayer(1,value));



        }

    }
}