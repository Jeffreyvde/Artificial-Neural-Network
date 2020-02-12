namespace NeuralNetworks.Neurons
{
    /// <summary>
    /// THis is the class designed for a basic Neuron. It only holds an activation value.
    /// </summary>
    public class BaseNeuron
    {
        public double Acitvation { get; private set; }

        /// <summary>
        /// Constructor for the base neuron
        /// </summary>
        /// <param name="activation"></param>
        public BaseNeuron(double activation)
        {
            Acitvation = activation;
        }
    }
}
