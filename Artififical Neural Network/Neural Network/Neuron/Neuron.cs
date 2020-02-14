using Newtonsoft.Json;
using System;

namespace NeuralNetwork.Neurons
{
    public abstract class Neuron : BaseNeuron, IBackpropogatable
    {
        public double Bias { get; private set; }
        public IActivation ActivationFunction { get; protected set; }

        //Backpropogation
        [JsonIgnore] public double DerivativeActivation { get; private set; }
        [JsonIgnore] public double DerivativeCost { get; protected set; }

        #region Initialization

        /// <summary>
        /// Constructor for Neuron class. That generates random bias between -1 and 1.
        /// </summary>
        /// <param name="activationFunction">The activation function for this Neuron</param>
        public Neuron(IActivation activationFunction)
        {
            ActivationFunction = activationFunction;
            do
            {
                Bias = Randomizer.Range(-1, 1);
            }
            while (Bias == 0);
        }

        /// <summary>
        /// Json constructor
        /// </summary>
        /// <param name="layerIndex"></param>
        /// <param name="layerRow"></param>
        /// <param name="bias"></param>
        [JsonConstructor()]
        public Neuron(double bias, IActivation activationFunction, double activation, Connection[] forwardConnections, Connection[] backwardsConnections) : base(activation, forwardConnections, backwardsConnections)
        {
            Bias = bias;
            ActivationFunction = activationFunction;
        }

        /// <summary>
        /// Set the required values of this neuron
        /// </summary>
        /// <param name="weightedSum"></param>
        /// <param name="activation"></param>
        public void SetValues(double weightedSum, double activation)
        {
            Activation = activation;
            DerivativeActivation = NeuralNetwork.activation.CalculateDerivativeActivation(weightedSum);
        }

        #endregion
        #region FeedForward

        /// <summary>
        /// Feedforward this Neuron
        /// </summary>
        public override void FeedForward()
        {
            throw new NotImplementedException();
        }

        #endregion
        #region Backpropogation
        /// <summary>
        /// Applies the gradient decent stap
        /// </summary>
        /// <param name="step"></param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            Bias -= learningRate * step;
        }
        #endregion
    }
}
