namespace NeuralNetwork.Backpropogation
{
    public interface IBackpropogatable
    {
        void ApplyGradientDecentStep(double step, double learningRate);
    }
}
