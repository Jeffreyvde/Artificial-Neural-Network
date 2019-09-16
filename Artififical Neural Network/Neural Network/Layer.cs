using System;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    public class Layer
    {
        public readonly int index;

        public Weight[,] weights;
        public readonly Neuron[] neurons;

        public Layer(int index, int sizeNeurons)
        {
            this.index = index;
            neurons = new Neuron[sizeNeurons];
        }

        /// <summary>
        /// Train this layer
        /// </summary>
        /// <param name="previousNeurons"></param>
        private void Train(Neuron[] previousNeurons)
        {
            Matrix<double> weigthMatrix = Converter.ConvertToMatrix(weights, neurons.Length, previousNeurons.Length);
            Vector<double> activations = Converter.ConvertToVector(previousNeurons, true);
            Vector<double> biases = Converter.ConvertToVector(neurons, false);

            Vector<double> weightedSum = weigthMatrix * activations + biases;
            InitializeNeuron(weightedSum, Sigmoid.CalculateSigmoid(weightedSum));
        }

        /// <summary>
        /// Initialize all neurons with weighted sum and activation
        /// </summary>
        /// <param name="weightedSum"></param>
        /// <param name="activation"></param>
        private void InitializeNeuron(Vector<double> weightedSum, Vector<double> activation)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].weightedSum = weightedSum[i];
                neurons[i].activation = activation[i];
            }
        }

        /// <summary>
        /// Generate an array of neurons
        /// </summary>
        /// <param name="random">Random generator for optimilization</param>
        public void GenerateNeurons(Random random)
        {
            for (int i = 0; i < neurons.Length; i++)
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
            weights = new Weight[neurons.Length, previousNeurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < previousNeurons.Length; j++)
                {
                    weights[i, j] = new Weight(index, neurons[i], previousNeurons[j], random);
                }
            }
        }
    }
}
