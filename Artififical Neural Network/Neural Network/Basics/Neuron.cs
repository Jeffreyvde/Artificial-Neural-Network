using System;
using Newtonsoft.Json;

namespace NeuralNetworks
{
    public class Neuron : IBackpropogatable
    {
        public readonly int layerIndex, layerRow;

        [JsonIgnore] public double activation;

        public double bias;
        [JsonIgnore] public double weightedSum;

        //Backpropogation
        [JsonIgnore] public double derivativeActivation;
        [JsonIgnore] public double derivativeCost;

        #region Initialization

        /// <summary>
        /// Constructor for Neuron class. That generates random bias between -1 and 1.
        /// </summary>
        /// <param name="layerIndex"></param>
        public Neuron(int layerIndex, int layerRow)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            bias = Randomizer.Range(-1, 1);
        }

        /// <summary>
        /// Constructor for Neuron class. That already has an activation.
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="random"></param>
        public Neuron(int layerIndex, int layerRow, float activation)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            this.activation = activation;
        }

        /// <summary>
        /// Json constructor
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="layerRow"></param>
        /// <param name="bias"></param>
        [JsonConstructor()]
        public Neuron(int layerIndex, int layerRow, double bias)
        {
            this.layerIndex = layerIndex;
            this.layerRow = layerRow;

            this.bias = bias;
        }

        /// <summary>
        /// Set the required values of this neuron
        /// </summary>
        /// <param name="weightedSum"></param>
        /// <param name="activation"></param>
        public void SetValues(double weightedSum, double activation)
        {
            this.weightedSum = weightedSum;
            this.activation = activation;

            derivativeActivation = NeuralNetwork.activation.CalculateDerivativeActivation(weightedSum);
        }

        #endregion
        #region Backpropogation


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
        public void BackPropogate(int traningData, GradientDescent gradientDecent)
        {
            derivativeCost = 2 * (activation - traningData);
            gradientDecent.Add(derivativeActivation * derivativeCost, this);
        }

        /// <summary>
        /// Backpropogate this hidden neuron
        /// </summary>
        /// <param name="nextLayer"></param>
        public void BackPropogate(Layer nextLayer, GradientDescent gradientDecent)
        {
            for (int i = 0; i < nextLayer.neurons.Length; i++)
            {
                Neuron neuron = nextLayer.neurons[i];
                derivativeCost += nextLayer.GetWeight(i, layerRow).weight * neuron.derivativeActivation * neuron.derivativeCost;
            }
            derivativeCost /= nextLayer.neurons.Length;

            gradientDecent.Add(derivativeActivation * derivativeCost, this);
        }

        /// <summary>
        /// Applies the gradient decent stap
        /// </summary>
        /// <param name="step"></param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            bias -= learningRate * step;
        }

        #endregion
    }
}
