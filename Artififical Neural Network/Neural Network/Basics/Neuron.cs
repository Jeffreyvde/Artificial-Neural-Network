using System;

namespace NeuralNetworks
{
    public class Neuron
    {
        public readonly int layerIndex, layerRow;

        public double activation;

        public double bias;
        public double weightedSum;

        //Backpropogation
        public double derivativeActivation;
        public double derivativeCost;
        public double derivativeBias;

        /// <summary>
        /// Constructor for Neuron class. That generates random bias between -1 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        public Neuron(int layerIndex, int layerRow)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            bias = Randomizer.GetRandomNumber(-1, 1);
        }

        /// <summary>
        /// Constructor for Neuron class. That already has an activation.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Neuron(int layerIndex, int layerRow, double activation)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            this.activation = activation;
        }

        /// <summary>
        /// Set the required values of this neuron
        /// </summary>
        /// <param name="weightedSum"></param>
        /// <param name="activation"></param>
        public void SetValues(double weightedSum, double activation, IActivation activationFunction)
        {
            this.weightedSum = weightedSum;
            this.activation = activation;

            derivativeActivation = activationFunction.CalculateDerivativeActivation(weightedSum);
        }

        /// <summary>
        /// Calculate the cost with training data
        /// </summary>
        /// <param name="trainingData">value from training data</param>
        /// <param name="backpropogate">Do you want the derivative value</param>
        /// <returns></returns>
        public double CalculateCost(double trainingData)
        {
            return Math.Pow(activation - trainingData, 2);
        }

        /// <summary>
        /// Backpropogate this output neuron
        /// </summary>
        /// <param name="traningData"></param>
        public void BackPropogate(int traningData)
        {
            derivativeCost = 2 * (activation - traningData);
            derivativeBias = derivativeActivation * derivativeCost;
        }

        /// <summary>
        /// Backpropogate this hidden neuron
        /// </summary>
        /// <param name="nextLayer"></param>
        public void BackPropogate(Layer nextLayer)
        {
            for (int i = 0; i < nextLayer.neurons.Length; i++)
            {
                Neuron neuron = nextLayer.neurons[i];
                derivativeCost += nextLayer.GetWeight(i, layerRow).weight * neuron.derivativeActivation * neuron.derivativeCost;
            }
            derivativeCost /= nextLayer.neurons.Length;
            derivativeBias = derivativeActivation * derivativeCost;
        }
    }
}
