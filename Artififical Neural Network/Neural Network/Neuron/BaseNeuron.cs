using Newtonsoft.Json;

namespace NeuralNetwork.Neurons
{
    /// <summary>
    /// This is the class designed for a basic Neuron. It only holds an activation value.
    /// </summary>
    public abstract class BaseNeuron
    {
        [JsonIgnore] public double Activation { get; protected set; }
        public Connection[] ForwardConnections { get; protected set; }
        public Connection[] BackwardsConnections { get; protected set; }

        /// <summary>
        /// Constructor for the base neuron
        /// </summary>
        /// <param name="activation"></param>
        public BaseNeuron(double activation)
        {
            Activation = activation;
        }


        /// <summary>
        /// Empty constructor only for child classes (Activation for certain neurons will be set later)
        /// </summary>
        protected BaseNeuron() { }

        /// <summary>
        /// Constructor used for json parsing
        /// </summary>
        /// <param name="activation"></param>
        /// <param name="forwardConnections"></param>
        /// <param name="backwardsConnections"></param>
        [JsonConstructor]
        public BaseNeuron(double activation, Connection[] forwardConnections, Connection[] backwardsConnections) : this(activation)
        {
            ForwardConnections = forwardConnections;
            BackwardsConnections = backwardsConnections;
        }

        /// <summary>
        /// Set the connection for this neuron
        /// </summary>
        /// <param name="forward"></param>
        /// <param name="backwards"></param>
        public void SetConnections(Connection[] forward, Connection[] backwards)
        {
            ForwardConnections = forward;
            BackwardsConnections = backwards;
        }

        /// <summary>
        /// Feedforward the Neuron
        /// </summary>
        public abstract void FeedForward();

        /// <summary>
        /// Backpropogate the Neuron
        /// </summary>
        public virtual void BackPropogate(GradientDescent descent)
        {
            for (int i = 0; i < ForwardConnections.Length; i++)
            {
                ForwardConnections[i].BackPropogate();
            }
        }
    }
}
