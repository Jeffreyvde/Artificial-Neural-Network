namespace NeuralNetwork
{
    public class Connection
    {
        public Neuron startNeuron;
        public Neuron endNeuron;

        public Connection(Neuron startNeuron, Neuron endNeuron)
        {
            this.startNeuron = startNeuron;
            this.endNeuron = endNeuron;
        }
    }
}
