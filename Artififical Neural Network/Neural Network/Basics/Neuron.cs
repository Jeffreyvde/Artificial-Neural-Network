using Newtonsoft.Json;
using System;
using System.Threading;

namespace NeuralNetworks
{
    public class Neuron : IBackpropogatable
    {
        public readonly int layerIndex, layerRow;
        public double bias;

        //Feedforward 
        [JsonIgnore]
        public double Activation {
            get { return feedforwardCache.Value.activation; }
            set {
                feedforwardCache = new ThreadLocal<FeedforwardCache>
                {
                    Value = new FeedforwardCache(value, 0)
                };
            }
        }


        //Backpropogation
        [JsonIgnore] public double DerivativeActivation { get { return feedforwardCache.Value.derivativeActivation; } }
        [JsonIgnore] public double DerivativeCost { get { return backpropCache.Value.derivativeCost; } }

        [JsonIgnore] private ThreadLocal<FeedforwardCache> feedforwardCache;
        [JsonIgnore] private ThreadLocal<BackpropCache> backpropCache;

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
            feedforwardCache = new ThreadLocal<FeedforwardCache>
            {
                Value = new FeedforwardCache(activation, NeuralNetwork.activation.CalculateDerivativeActivation(weightedSum))
            };
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
            return Math.Pow(Activation - trainingData, 2);
        }

        /// <summary>
        /// Backpropogate this output neuron
        /// </summary>
        /// <param name="traningData"></param>
        public void BackPropogate(int traningData, GradientDescent gradientDecent)
        {
            double derivativeCost = 2 * (Activation - traningData);

            SetBackPropCache(derivativeCost);
            gradientDecent.Add(DerivativeActivation * DerivativeCost, this);
        }

        /// <summary>
        /// Backpropogate this hidden neuron
        /// </summary>
        /// <param name="nextLayer"></param>
        public void BackPropogate(Layer nextLayer, GradientDescent gradientDecent)
        {
            double derivativeCost = 0;
            for (int i = 0; i < nextLayer.neurons.Length; i++)
            {
                Neuron neuron = nextLayer.neurons[i];
                derivativeCost += nextLayer.GetWeight(i, layerRow).weight * neuron.DerivativeActivation * neuron.DerivativeCost;
            }
            derivativeCost /= nextLayer.neurons.Length;

            SetBackPropCache(derivativeCost);
            gradientDecent.Add(DerivativeActivation * DerivativeCost, this);
        }

        /// <summary>
        /// Applies the gradient decent stap
        /// </summary>
        /// <param name="step"></param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            bias -= learningRate * step;
        }

        private void SetBackPropCache(double derivativeCost)
        {
            backpropCache = new ThreadLocal<BackpropCache>
            {
                Value = new BackpropCache(derivativeCost)
            };
        }

        #endregion
    }

    /// <summary>
    /// Cache used by the Neuron class
    /// </summary>
    public struct BackpropCache
    {
        public double derivativeCost;

        public BackpropCache(double derivativeCost)
        {
            this.derivativeCost = derivativeCost;
        }
    }

    /// <summary>
    /// Feedforward cache values
    /// </summary>
    public struct FeedforwardCache
    {
        public double activation;
        public double derivativeActivation;

        public FeedforwardCache(double activation, double derivativeActivation)
        {
            this.activation = activation;
            this.derivativeActivation = derivativeActivation;
        }
    }


}
