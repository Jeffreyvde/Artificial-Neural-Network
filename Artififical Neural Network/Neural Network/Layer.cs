using System;

namespace NeuralNetwork
{
    public class Layer
    {
        public readonly int index;

        public readonly Weight[,] weights;
        public readonly Neuron[] neurons;

        public Layer(int index, int sizeNeurons)
        {
            this.index = index;
            neurons = new Neuron[sizeNeurons];

        }

        /// <summary>
        /// Generate an array of neurons
        /// </summary>
        /// <param name="random">Random generator for optimilization</param>
        public void GenerateNeurons(Random random)
        {
            for(int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(index, i, random);
            }
        }

        /// <summary>
        /// Generate weights for this layer
        /// </summary>
        /// <param name="previousNeurons">previous neurons to create Connection</param>
        /// <param name="random">Random generator for optimilization</param>
        public void GenerateWeights(Neuron[] previousNeurons, Random random)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; i < previousNeurons.Length; j++)
                {
                    weights[i, j] = new Weight(index, neurons[i], previousNeurons[i], random);
                }
            }
        }
    }
}
