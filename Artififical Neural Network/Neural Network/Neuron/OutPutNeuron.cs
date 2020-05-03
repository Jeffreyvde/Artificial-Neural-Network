using System;
using NeuralNetwork.Activations;
using NeuralNetwork.Backpropogation;
using NeuralNetwork.Utilities;

namespace NeuralNetwork.Neurons
{
    [Serializable]
    public class OutPutNeuron : Neuron
    {
        [NonSerialized]  private double correctOutput = float.MinValue;

        /// <summary>
        /// Default constructor
        /// </summary>
        /// <param name="activationFunction">The activation function to be used</param>
        public OutPutNeuron(IActivation activationFunction) : base(activationFunction) { }

        /// <summary>
        /// Constructor with random is injected
        /// </summary>
        /// <param name="random">The random class to be used</param>
        /// <param name="activationFunction">The activation function to be used</param>
        public OutPutNeuron(IRandom random, IActivation activationFunction) : base(random, activationFunction) { }

        /// <summary>
        /// Set the test data 
        /// </summary>
        /// <param name="correctOutput">The correct output for this variable</param>
        public void SetOutput(double correctOutput)
        {
            this.correctOutput = correctOutput;
        }

        /// <summary>
        /// Calculates the backpropagation values from the cost of the training data
        /// </summary>
        public override void BackPropagate(GradientDescent descent)
        {
            if (Math.Abs(correctOutput - float.MinValue) < 0.1)
                throw new InvalidOperationException("Set the correct output before backpropagation this Neuron.");

            if(descent == null)
                throw new ArgumentNullException(nameof(descent));

            DerivativeCost = 2 * (Activation - correctOutput);
            descent.Add(DerivativeActivation * DerivativeCost, this);
        }
    }
}
