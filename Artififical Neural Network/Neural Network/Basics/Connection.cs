using Newtonsoft.Json;

namespace NeuralNetwork.Neurons
{
    public class Connection
    {
        public Neuron StartNeuron { get; private set; }
        public Neuron EndNeuron { get; private set; }

        public double Weight { get { return weight.weight; } }

        private readonly Weight weight;

        /// <summary>
        /// Constructor for a connection between Neurons
        /// </summary>
        /// <param name="startNeuron"></param>
        /// <param name="endNeuron"></param>
        public Connection(Neuron startNeuron, Neuron endNeuron)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;

            weight = new Weight();
        }

        /// <summary>
        /// Json constructor
        /// </summary>
        /// <param name="startNeuron"></param>
        /// <param name="endNeuron"></param>
        /// <param name="weight"></param>
        [JsonConstructor]
        public Connection(Neuron startNeuron, Neuron endNeuron, Weight weight)
        {
            StartNeuron = startNeuron;
            EndNeuron = endNeuron;
            this.weight = weight;
        }

        public void BackPropogate()
        {
            throw new System.NotImplementedException();
        }
    }
}
