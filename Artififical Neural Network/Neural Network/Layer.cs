using System;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public class Layer
    {
        public readonly int index;

        public Weight[,] weights;
        public readonly Neuron[] neurons;
        private readonly IActivation activationFunction;

        #region Initialization
        public Layer(int index, int sizeNeurons, IActivation activationFunction)
        {
            this.index = index;
            this.activationFunction = activationFunction;

            neurons = new Neuron[sizeNeurons];
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(index, i);
            }
        }

        /// <summary>
        /// Generate weights for this layer
        /// </summary>
        /// <param name="previousNeurons">previous neurons to create Connection</param>
        public void GenerateWeights(Neuron[] previousNeurons)
        {
            weights = new Weight[neurons.Length, previousNeurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < previousNeurons.Length; j++)
                {
                    weights[i, j] = new Weight(index, neurons[i], previousNeurons[j]);
                }
            }
        }
        #endregion
        #region Training

        /// <summary>
        /// Train this layer
        /// </summary>
        /// <param name="previousNeurons"></param>
        public void Train(Neuron[] previousNeurons)
        {
            Matrix<double> weigthMatrix = Converter.ConvertToMatrix(weights, neurons.Length, previousNeurons.Length);
            Vector<double> activations = Converter.ConvertToVector(previousNeurons, true);
            Vector<double> biases = Converter.ConvertToVector(neurons, false);

            Vector<double> weightedSum = weigthMatrix * activations + biases;

            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].SetValues(weightedSum[i], activations[i], activationFunction);
            }
        }

        #endregion
        #region Backpropogation

        /// <summary>
        /// Backpropogate this output layer
        /// </summary>
        /// <param name="correctOutputNeuron"></param>
        public void BackPropogate(int correctOutputNeuron)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].BackPropogate(correctOutputNeuron == i ? 1 : 0);
            }
        }

        /// <summary>
        /// Backpropogate this hidden layer
        /// </summary>
        /// <param name="nextLayer"></param>
        public void BackPropogate(Layer nextLayer)
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i].BackPropogate(nextLayer);
            }
        }

        /// <summary>
        /// Get a weight
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public Weight GetWeight(int row, int column)
        {
            return weights[row, column];
        }

        #endregion


    }
}
