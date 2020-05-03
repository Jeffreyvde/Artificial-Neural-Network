using NeuralNetwork.Neurons;
using NeuralNetwork.Activations;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Layers
{
    /// <summary>
    /// Dense layer the basic building block for a Neural Network
    /// </summary>
    [System.Serializable]
    public class DenseLayer : Layer
    {
        /// <summary>
        /// Initialise the Layer with Hidden neurons
        /// </summary>
        /// <param name="size">The size of this layer</param>
        /// <param name="activationFunction">The function to be used for the activation calculations</param>
        public DenseLayer(int size, IActivation activationFunction)
        {
            Neurons = new BaseNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new HiddenNeuron(activationFunction);
            }
        }

        /// <summary>
        /// Initialise the Layer with Hidden neurons with custom random method. Mostly used for testing.
        /// </summary>
        /// <param name="size">The size of this layer</param>
        /// <param name="random">The custom random value</param>
        /// <param name="activationFunction">The function to be used for the activation calculations</param>
        public DenseLayer(int size, IRandom random, IActivation activationFunction)
        {
            Neurons = new BaseNeuron[size];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new HiddenNeuron(random, activationFunction);
            }
        }
    }
}
