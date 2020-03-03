using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    public abstract class Neuron : BaseNeuron, IBackpropogatable
    {

        public double Bias { get; private set; }
        public IActivation ActivationFunction { get; protected set; }

        //Backpropogation
        [System.NonSerialized] private double derivativeActivation;
        [System.NonSerialized] private double derivativeCost;

        public double DerivativeActivation { get => derivativeActivation; private set => derivativeActivation = value; }
        public double DerivativeCost { get => derivativeCost; protected set => derivativeCost = value; }

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
                Bias = Random.Range(-1, 1);
            }
            while (Bias == 0);
        }
        #endregion
        #region FeedForward

        /// <summary>
        /// Feedforward this Neuron
        /// </summary>
        public override void FeedForward()
        {
            double weightedSum = Converter.GetWeights(BackwardsConnections) * Converter.GetStartActivations(BackwardsConnections) * Bias;
            Activation = ActivationFunction.CalculateActivation(weightedSum);
            DerivativeActivation = ActivationFunction.CalculateDerivativeActivation(weightedSum);
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
