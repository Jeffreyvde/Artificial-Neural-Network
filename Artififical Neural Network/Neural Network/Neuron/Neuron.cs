using System;
using NeuralNetwork.Utilities;
using NeuralNetwork.Backpropogation;
using NeuralNetwork.Activations;

namespace NeuralNetwork.Neurons
{
    [Serializable]
    public abstract class Neuron : BaseNeuron, IBackpropogatable
    {

        public double Bias { get; private set; }
        public IActivation ActivationFunction { get; protected set; }

        // Backpropagation
        [NonSerialized] private double derivativeActivation;
        [NonSerialized] private double derivativeCost;

        public double DerivativeActivation { get => derivativeActivation; private set => derivativeActivation = value; }
        public double DerivativeCost { get => derivativeCost; protected set => derivativeCost = value; }

        #region Initialization

        /// <summary>
        /// Constructor for Neuron class. Where both random and activation can be passed as a dependency.
        /// </summary>
        /// <param name="activationFunction">The activation function for this Neuron</param>
        /// <param name="random">The way random start value was created</param>
        protected Neuron(IRandom random, IActivation activationFunction)
        {
            ActivationFunction = activationFunction;

            if (random == null)
                throw new ArgumentNullException(nameof(random));
            Bias = random.Range(-1, 1);
        }

        /// <summary>
        /// Constructor for Neuron class. That generates random bias between -1 and 1.
        /// </summary>
        /// <param name="activationFunction">The activation function for this Neuron</param>
        /// <parm name="random">The way random start value was created</parm>
        protected Neuron(IActivation activationFunction) : this(new Utilities.RandomRange(), activationFunction)
        {
        }

        #endregion
        #region FeedForward

        /// <summary>
        /// Feed forward this Neuron
        /// </summary>
        public override void FeedForward()
        {
            if(BackwardsConnections == null)
                throw new InvalidOperationException($"{nameof(BackwardsConnections)} need to be initialed before feed forwarding.");

            double weightedSum = Converter.GetWeights(BackwardsConnections) * Converter.GetStartActivations(BackwardsConnections) * Bias;
            Activation = ActivationFunction.CalculateActivation(weightedSum);
            DerivativeActivation = ActivationFunction.CalculateDerivativeActivation(weightedSum);
        }

        #endregion
        #region Backpropogation
        /// <summary>
        /// Applies the gradient decent step
        /// </summary>
        /// <param name="step">the step</param>
        /// <param name="learningRate">The learning rate that will be applied to the step</param>
        public void ApplyGradientDecentStep(double step, double learningRate)
        {
            Bias -= learningRate * step;
        }
        #endregion
    }
}
