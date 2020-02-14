namespace NeuralNetwork.Neurons
{
    public class InputNeuron : BaseNeuron
    {
        public InputNeuron(double activation) : base(activation){}
        public InputNeuron() { }

        /// <summary>
        /// Set the activation function
        /// </summary>
        /// <param name="activation"></param>
        public void SetActivation(double activation)
        {
            Activation = activation;
        }

        /// <summary>
        /// WARNING Can no be called for the input Neuron
        /// </summary>
        public override void FeedForward()
        {
            throw new System.NotImplementedException("The input layer can not be Feedforwarded.");
        }
    }
}
