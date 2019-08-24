class NeuNet {
    constructor(
        ilayer,
        hlayer,
        Hhlayer,
        olayer,
        weight_ih = null,
        hbias = null,
        weight_hh = null,
        Hhbias = null,
        weight_ho = null,
        obias = null
    ) {
        if (ilayer instanceof NeuNet) {
            let lyr = ilayer;
            this.input_nodes = lyr.input_nodes;
            this.hidden_nodes = lyr.hidden_nodes;
            this.Hhidden_nodes = lyr.Hhidden_nodes;
            this.output_nodes = lyr.output_nodes;

            this.weight_ih = lyr.weight_ih.copy();
            this.weight_hh = lyr.weight_hh.copy();
            this.weight_ho = lyr.weight_ho.copy();

            this.hbias = lyr.hbias.copy();
            this.Hhbias = lyr.Hhbias.copy();
            this.obias = lyr.obias.copy();
        } else {
            this.input_nodes = ilayer;
            this.hidden_nodes = hlayer;
            this.Hhidden_nodes = Hhlayer;
            this.output_nodes = olayer;

            this.weight_ih = new Matrix(this.hidden_nodes, this.input_nodes);
            this.weight_hh = new Matrix(this.Hhidden_nodes, this.hidden_nodes);
            this.weight_ho = new Matrix(this.output_nodes, this.Hhidden_nodes);

            this.hbias = new Matrix(this.hidden_nodes, 1);
            this.Hhbias = new Matrix(this.Hhidden_nodes, 1);
            this.obias = new Matrix(this.output_nodes, 1);

            if (weight_ih && weight_ho && weight_hh) {
                let wih = Matrix.subtract_array(weight_ih, this.hidden_nodes, this.input_nodes);
                let bih = Matrix.fromArray(hbias);

                let whh = Matrix.subtract_array(weight_hh, this.Hhidden_nodes, this.hidden_nodes);
                let bhh = Matrix.fromArray(Hhbias);

                let who = Matrix.subtract_array(weight_ho, this.output_nodes, this.Hhidden_nodes);
                let bho = Matrix.fromArray(obias);

                this.weight_ih = wih;
                this.weight_hh = whh;
                this.weight_ho = who;
                this.hbias = bih;
                this.Hhbias = bhh;
                this.obias = bho;
            } else {
                this.weight_ih.randomize();
                this.weight_hh.randomize();
                this.weight_ho.randomize();

                this.hbias.randomize();
                this.Hhbias.randomize();
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
        hidden.map(this.ActFunc.func);

        let Hhidden = Matrix.multiply(this.weight_hh, hidden);
        Hhidden.add(this.Hhbias);
        Hhidden.map(this.ActFunc.func);

        // Generating the output's output!
        let output = Matrix.multiply(this.weight_ho, Hhidden);
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
        hidden.map(this.ActFunc.func);

        let Hhidden = Matrix.multiply(this.weight_hh, hidden);
        Hhidden.add(this.Hhbias);
        Hhidden.map(this.ActFunc.func);

        // Generating the output's output!
        let outputs = Matrix.multiply(this.weight_ho, Hhidden);
        outputs.add(this.obias);
        outputs.map(this.ActFunc.func);


        // Convert array to matrix object
        let targets = Matrix.fromArray(target_array);

        // Calculate the error ==> ERROR = TARGETS - OUTPUTS
        let output_errors = Matrix.subtract(targets, outputs); // console.log(output_errors)

        // let gradient = outputs * (1 - outputs);
        let gradients = Matrix.map(outputs, this.ActFunc.dfunc);
        gradients.multiply(output_errors);
        gradients.multiply(this.LearningRate);

        // Calculate deltas
        let Hhidden_T = Matrix.transpose(Hhidden);
        let weight_ho_deltas = Matrix.multiply(gradients, Hhidden_T);

        // Adjust the weights by deltas
        this.weight_ho.add(weight_ho_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.obias.add(gradients);

        // Calculate the hidden layer errors
        let who_t = Matrix.transpose(this.weight_ho);
        let Hhidden_errors = Matrix.multiply(who_t, output_errors);

        // Calculate hidden gradient
        let Hhidden_gradient = Matrix.map(Hhidden, this.ActFunc.dfunc);
        Hhidden_gradient.multiply(Hhidden_errors);
        Hhidden_gradient.multiply(this.LearningRate);


        // Calculate deltas
        let hidden_T = Matrix.transpose(hidden);
        let weight_hh_deltas = Matrix.multiply(Hhidden_gradient, hidden_T);

        // Adjust the weights by deltas
        this.weight_hh.add(weight_hh_deltas);
        // Adjust the bias by its deltas (which is just the gradients)
        this.Hhbias.add(Hhidden_gradient);

        // Calculate the hidden layer errors
        let whh_t = Matrix.transpose(this.weight_hh);
        let hidden_errors = Matrix.multiply(whh_t, output_errors);

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

        $("#w1").val(this.weight_ih.data[0][0].toFixed(4));
        $("#w2").val(this.weight_ih.data[0][1].toFixed(4));
        $("#w3").val(this.weight_ih.data[1][0].toFixed(4));
        $("#w4").val(this.weight_ih.data[1][1].toFixed(4));
        $("#w5").val(this.weight_hh.data[0][0].toFixed(4));
        $("#w6").val(this.weight_hh.data[0][1].toFixed(4));
        $("#w7").val(this.weight_ho.data[0][0].toFixed(4));

        // console.log(gradients.data[0][0].toFixed(4))


        return output_errors.toArray();
    }



    serialize() {
        return JSON.stringify(this);
    }



    static deserialize(data) {
        if (typeof data == 'string') {
            data = JSON.parse(data);
        }
        let nn = new NeuNet(
            data.input_nodes, data.hidden_nodes, data.Hhidden_nodes, data.output_nodes
        );
        nn.weight_ih = Matrix.deserialize(data.weight_ih);
        nn.weight_hh = Matrix.deserialize(data.weight_hh);
        nn.weight_ho = Matrix.deserialize(data.weight_ho);
        nn.hbias = Matrix.deserialize(data.hbias);
        nn.Hhbias = Matrix.deserialize(data.Hhbias);
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
        this.weight_hh.map(func);
        this.weight_ho.map(func);
        this.hbias.map(func);
        this.Hhbias.map(func);
        this.obias.map(func);
    }
}




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
