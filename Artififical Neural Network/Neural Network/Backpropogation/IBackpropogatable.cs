namespace NeuralNetworks
{
    public interface IBackpropogatable
    {
        void ApplyGradientDecentStep(double step);
    }
}
