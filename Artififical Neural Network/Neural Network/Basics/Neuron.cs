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
        /// <param name="derivative">Do you want the derivative value</param>
        /// <returns></returns>
        public double CalculateCost(double trainingData, bool derivative = false)
        {
            if (!derivative)
                return Math.Pow(activation - trainingData, 2);

            derivativeCost = 2 * (activation - trainingData);
            return derivativeCost;
        }
    }
}
