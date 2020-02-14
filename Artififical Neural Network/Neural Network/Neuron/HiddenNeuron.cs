namespace NeuralNetwork.Neurons
{
    public class HiddenNeuron : Neuron
    {
        /// <summary>
        /// Default constructor for the hidden neuron
        /// </summary>
        /// <param name="activationFunction"></param>
        public HiddenNeuron(IActivation activationFunction) : base(activationFunction) { }

        /// <summary>
        /// Backpropogate this hidden neuron
        /// </summary>
        /// <param name="nextLayer"></param>
        public override void BackPropogate(GradientDescent descent)
        {
            Neuron nextNeuron;
            for (int i = 0; i < ForwardConnections.Length; i++)
            {
                nextNeuron = ForwardConnections[i].EndNeuron;
                DerivativeCost += ForwardConnections[i].Weight * nextNeuron.DerivativeActivation * nextNeuron.DerivativeActivation;
            }
            DerivativeCost /= ForwardConnections.Length;

            descent.Add(DerivativeActivation * DerivativeCost, this);
        }
    }
}
