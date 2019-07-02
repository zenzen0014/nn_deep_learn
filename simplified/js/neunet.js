class ActivationFunction{
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}
let sigmoid = new ActivationFunction(
    a => 1 / (1 + Math.exp(-a)),
    b => b * (1 - b)
);


class NeuNet {
    constructor(ilayer, hlayer, olayer, wih = null, who = null, bh = null, bo = null) {
      if(ilayer instanceof NeuNet){
        let lyr = ilayer;
        this.input_nodes = lyr.input_nodes;
        this.hidden_nodes = lyr.hidden_nodes;
        this.output_nodes = lyr.output_nodes;

        this.weight_ih = lyr.weight_ih.copy()
        this.weight_ho = lyr.weight_ho.copy()
        this.hbias = lyr.hbias.copy()
        this.obias = lyr.obias.copy()
      }else{
        this.input_nodes = ilayer;
        this.hidden_nodes = hlayer;
        this.output_nodes = olayer;

        this.weight_ih = new Matrix(this.hidden_nodes, this.input_nodes)
        this.weight_ho = new Matrix(this.output_nodes, this.hidden_nodes)
        
        this.hbias = new Matrix(this.hidden_nodes, 1)
        this.obias = new Matrix(this.output_nodes, 1)

        if(wih && who){
            this.weight_ih = Matrix.subtract_array(wih, this.hidden_nodes, this.input_nodes)
            this.weight_ho = Matrix.subtract_array(who, this.output_nodes, this.hidden_nodes)
            
            this.hbias = Matrix.fromArray(bh);
            this.obias = Matrix.fromArray(bo);
        }else{
            this.weight_ih.randomize();
            this.weight_ho.randomize();
            this.hbias.randomize();
            this.obias.randomize();
        }
        console.log(this.weight_ih)
      }
      this.setLearningRate();
      this.setActivationFunction();
    }
    setLearningRate(lr = 0.3){
      this.LearningRate = lr
    }
    setActivationFunction(af = sigmoid){
      this.ActFunc = af
    }
}