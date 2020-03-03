using System;
using NeuralNetwork.Backpropogation;

namespace NeuralNetwork.Neurons
{
    /// <summary>
    /// This is the class designed for a basic Neuron. It only holds an activation value.
    /// </summary>
    [Serializable]
    public abstract class BaseNeuron
    {
        [NonSerialized] private double activation;

        public double Activation { get { return activation; } protected set { activation = value; } }
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
        /// Generate the connections for this neuron
        /// </summary>
        /// <param name="nextlayer"></param>
        /// <param name="previousLayer"></param>
        public void EstablishConnections(BaseNeuron[] nextlayer, BaseNeuron[] previousLayer)
        {
            if (nextlayer == null) ForwardConnections = null;
            else
            {
                ForwardConnections = new Connection[nextlayer.Length];
                for (int i = 0; i < nextlayer.Length; i++)
                {
                    ForwardConnections[i] = new Connection(this, nextlayer[i]);
                }
            }
            if (previousLayer == null) BackwardsConnections = null;
            else
            {
                BackwardsConnections = new Connection[previousLayer.Length];
                for (int i = 0; i < BackwardsConnections.Length; i++)
                {
                    BackwardsConnections[i] = new Connection(previousLayer[i], this);
                }
            }
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
                ForwardConnections[i].BackPropogate(descent);
            }
        }
    }
}
