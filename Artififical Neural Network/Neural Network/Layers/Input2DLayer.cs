using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Layers
{
    public class Input2DLayer : InputLayer
    {
        public Input2DLayer(int length, int width) : base(length * width) { }

        public void SetInput(double[,] activation)
        {
            base.SetInput(null);
        }
    }
}
