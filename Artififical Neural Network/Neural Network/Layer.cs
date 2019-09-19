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

        public Layer(int index, int sizeNeurons, IActivation activationFunction)
        {
            this.index = index;
            neurons = new Neuron[sizeNeurons];
            this.activationFunction = activationFunction;
        }

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
            InitializeNeuron(weightedSum, activationFunction.CalculateActivation(weightedSum));
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
                neurons[i].SetValues(weightedSum[i], activation[i], activationFunction);
            }
        }

        /// <summary>
        /// Generate an array of neurons
        /// </summary>
        public void GenerateNeurons()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(index, i);
            }
        }

        /// <summary>
        /// Assing an array of neurons with input data
        /// </summary>
        public void AssingNeurons(double[] input)
        {
            if (input.Length != neurons.Length) throw new Exception("Input not equal to neurons lenght");

            for (int i = 0; i < neurons.Length; i++)
            {
                neurons[i] = new Neuron(index, i, input[i]);
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
        #region Backpropogation

        /// <summary>
        /// Calculate the cost from training data
        /// </summary>
        /// <param name="correctValue">Wich neuron is correct. (Starts at zero)</param>
        /// <param name="derivative">Do you need the derivative cost</param>
        /// <returns></returns>
        public double[] CalculateCost(int correctValue,  bool derivative = false)
        {
            double[] values = new double[neurons.Length];
            for (int i = 0; i < neurons.Length; i++)
            {
                values[i] = neurons[i].CalculateCost(correctValue == i ? 1 : 0, derivative);
            }
            return values;
        }

        public void CalculateCostDerivatives()
        {
            for (int i = 0; i < neurons.Length; i++)
            {
                
            }
        }

        #endregion


    }
}
