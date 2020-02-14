using System;
using Newtonsoft.Json;

namespace NeuralNetwork.Neurons
{
    public class OutPutNeuron : Neuron
    {
        [JsonIgnore] private float correctOuput = float.MinValue;

        /// <summary>
        /// Default constructor
        /// </summary>
        /// <param name="activationFunction"></param>
        public OutPutNeuron(IActivation activationFunction) : base(activationFunction) { }

        /// <summary>
        /// Feedforward the output neuron. This calculates its Activation.
        /// </summary>
        public override void FeedForward()
        {
            throw new System.NotImplementedException();
        }

        /// <summary>
        /// Set the test data 
        /// </summary>
        /// <param name="testData"></param>
        public void SetTestData(float correctOuput)
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
