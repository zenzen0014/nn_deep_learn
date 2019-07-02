// Other techniques for learning

  class ActivationFunction {
    constructor(func, dfunc) {
      this.func = func;
      this.dfunc = dfunc;
    }
  }

  let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
  );

  let tanh = new ActivationFunction(
    x => Math.tanh(x),
    y => 1 - (y * y)
  );


  class NeuNet {
    /*
    * if first argument is a NeuNet the constructor clones it
    * USAGE: cloned_nn = new NeuNet(to_clone_nn);
    */
    constructor(ilayer, hlayer, olayer,weight_ih=null, hbias=null, weight_ho=null, obias=null) {
      if (ilayer instanceof NeuNet) {
        let lyr = ilayer;
        this.input_nodes = lyr.input_nodes;
        this.hidden_nodes = lyr.hidden_nodes;
        this.output_nodes = lyr.output_nodes;

        this.weight_ih = lyr.weight_ih.copy();
        this.weight_ho = lyr.weight_ho.copy();

        this.hbias = lyr.hbias.copy();
        this.obias = lyr.obias.copy();
      } else {
        this.input_nodes = ilayer;
        this.hidden_nodes = hlayer;
        this.output_nodes = olayer;

        this.weight_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weight_ho = new Matrix(this.output_nodes, this.hidden_nodes);

        this.hbias = new Matrix(this.hidden_nodes, 1);
        this.obias = new Matrix(this.output_nodes, 1);


        if(weight_ih && weight_ho){
          let wih = Matrix.subtract_array(weight_ih, this.hidden_nodes, this.input_nodes);
          let bih = Matrix.fromArray(hbias);
          let who = Matrix.subtract_array(weight_ho, this.output_nodes, this.hidden_nodes);
          let bho = Matrix.fromArray(obias);
          this.weight_ih = wih;
          this.weight_ho = who;
          this.hbias     = bih;
          this.obias     = bho;
        }else{
          this.weight_ih.randomize();
          this.weight_ho.randomize();

          this.hbias.randomize();
          this.obias.randomize();
        }

      }
      // TODO: copy these as well
      this.setLearningRate();
      this.setActivationFunction();
    }

    setLearningRate(LearningRate = 0.1) {
      this.LearningRate = LearningRate;
    }

    setActivationFunction(func = sigmoid) {
      this.ActFunc = func;
    }

    prediction(input_array) {
      // Generating the Hidden Outputs
      let inputs = Matrix.fromArray(input_array);

      let hidden = Matrix.multiply(this.weight_ih, inputs);
      hidden.add(this.hbias);
      // activation function!
      hidden.map(this.ActFunc.func);

      // Generating the output's output!
      let output = Matrix.multiply(this.weight_ho, hidden);
      output.add(this.obias);
      output.map(this.ActFunc.func);

      // Sending back to the caller!
      return output.toArray();
    }



    train(input_array, target_array) {
      // Generating the Hidden Outputs
      let inputs = Matrix.fromArray(input_array);
      let hidden = Matrix.multiply(this.weight_ih, inputs);
      hidden.add(this.hbias);

      // activation function!
      hidden.map(this.ActFunc.func);

      // Generating the output's output!
      let outputs = Matrix.multiply(this.weight_ho, hidden);
      outputs.add(this.obias);
      outputs.map(this.ActFunc.func);

      // Convert array to matrix object
      let targets = Matrix.fromArray(target_array);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      let output_errors = Matrix.subtract(targets, outputs);
      // console.log(output_errors)

      // let gradient = outputs * (1 - outputs);
      // Calculate gradient
      let gradients = Matrix.map(outputs, this.ActFunc.dfunc);
      gradients.multiply(output_errors);
      gradients.multiply(this.LearningRate);

      // Calculate deltas
      let hidden_T = Matrix.transpose(hidden);
      let weight_ho_deltas = Matrix.multiply(gradients, hidden_T);

      // Adjust the weights by deltas
      this.weight_ho.add(weight_ho_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      this.obias.add(gradients);

      // Calculate the hidden layer errors
      let who_t = Matrix.transpose(this.weight_ho);
      let hidden_errors = Matrix.multiply(who_t, output_errors);

      // Calculate hidden gradient
      let hidden_gradient = Matrix.map(hidden, this.ActFunc.dfunc);
      hidden_gradient.multiply(hidden_errors);
      hidden_gradient.multiply(this.LearningRate);

      // Calcuate input->hidden deltas
      let inputs_T = Matrix.transpose(inputs);
      let weight_ih_deltas = Matrix.multiply(hidden_gradient, inputs_T);

      this.weight_ih.add(weight_ih_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      this.hbias.add(hidden_gradient);

      // outputs.print();
      // targets.print();
      // gradients.print();//errror
      return output_errors.toArray();
    }



    serialize() {
      return JSON.stringify(this);
    }



    static deserialize(data) {
      if (typeof data == 'string') {
        data = JSON.parse(data);
      }
      let nn = new NeuNet(data.input_nodes, data.hidden_nodes, data.output_nodes);
      nn.weight_ih = Matrix.deserialize(data.weight_ih);
      nn.weight_ho = Matrix.deserialize(data.weight_ho);
      nn.hbias = Matrix.deserialize(data.hbias);
      nn.obias = Matrix.deserialize(data.obias);
      nn.LearningRate = data.LearningRate;
      return nn;
    }


    // Adding function for neuro-evolution
    copy() {
      return new NeuNet(this);
    }


    // Accept an arbitrary function for mutation
    mutate(func) {
      this.weight_ih.map(func);
      this.weight_ho.map(func);
      this.hbias.map(func);
      this.obias.map(func);
    }



  }
