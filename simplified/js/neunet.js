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
    setActivationFunction(func = sigmoid) {
      this.ActFunc = func
    }

    prediction(input_array){
      let inputs = Matrix.fromArray(input_array);
      let hidden = Matrix.multiply(this.weight_ih, inputs);
      hidden.add(this.hbias);
      hidden.map(this.ActFunc.func);

      let outputs = Matrix.multiply(this.weight_ho, hidden);
      outputs.add(this.obias);
      outputs.map(this.ActFunc.func);

      return outputs.toArray();
    }

    train(input_array, target_array){
      let inputs = Matrix.fromArray(input_array);
      let hidden = Matrix.multiply(this.weight_ih, inputs);
      hidden.add(this.hbias);
      hidden.map(this.ActFunc.func);

      let outputs = Matrix.multiply(this.weight_ho, hidden);
      outputs.add(this.obias);
      outputs.map(this.ActFunc.func);

      let targets = Matrix.fromArray(target_array);
      let outputs_error = Matrix.subtract(targets, outputs)
      console.log(outputs_error); //total error per epoch

      let gradients = Matrix.map(outputs, this.ActFunc.dfunc);
      gradients.multiply(outputs_error);
      gradients.multiply(this.LearningRate);

      let hidden_T = Matrix.transpose(hidden);
      let weight_ho_delta = Matrix.multiply(gradients, hidden_T)
      
      this.weight_h.add(weight_ho_delta);
      this.obias.add(gradients);

      let who_T = Matrix.transpose(this.weight_h);
      let hidden_errors = Matrix.multiply(who_T, outputs_error)

      let hgradient = Matrix.map(hidden, this.ActFunc.dfunc);
      hgradient.multiply(hidden_errors);
      hgradient.multiply(this.LearningRate);

      let input_T = Matrix.transpose(inputs);
      let weight_ih_delta = Matrix.multiply(hgradient, input_T)

      this.weight_ih.add(weight_ih_delta);
      this.hbias.add(hgradient);

      gradients.print();
      outputs.print();
      targets.print();
    }
}