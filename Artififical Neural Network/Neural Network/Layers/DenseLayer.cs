using NeuralNetwork.Neurons;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Dense layer the basic building block for a Neural Network
    /// </summary>
    public class DenseLayer : Layer<HiddenNeuron>
    {
        /// <summary>
        /// Initialises the Layer with Hidden neurons
        /// </summary>
        /// <param name="size"></param>
        /// <param name="activatFunction"></param>
        public DenseLayer(int size, IActivation activatFunction)
        {
            Neurons = new HiddenNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new HiddenNeuron(activatFunction);
            }
        }
    }
}
