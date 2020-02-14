namespace NeuralNetwork
{
    public interface IBackpropogatable
    {
        void ApplyGradientDecentStep(double step, double learningRate);
    }
}
