using System;

namespace NeuralNetwork.Neurons
{
    public class OutPutNeuron : Neuron
    {
        [NonSerialized]  private double correctOuput = float.MinValue;

        /// <summary>
        /// Default constructor
        /// </summary>
        /// <param name="activationFunction"></param>
        public OutPutNeuron(IActivation activationFunction) : base(activationFunction) { }

        /// <summary>
        /// Set the test data 
        /// </summary>
        /// <param name="testData"></param>
        public void SetOutput(double correctOuput)
        {
            this.correctOuput = correctOuput;
        }

        /// <summary>
        /// Calculates the backpropogation values from the cost of the trainingdat
        /// </summary>
        public override void BackPropogate(GradientDescent descent)
        {
            if (correctOuput == float.MinValue)
                throw new InvalidOperationException("Set correct ouput needs to be called before backprop");

            DerivativeCost = 2 * (Activation - correctOuput);
            descent.Add(DerivativeActivation * DerivativeCost, this);
        }
    }
}
