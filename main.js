const SCALE = 4.0
var canvas1 = document.getElementById("biological");
var canvas2 = document.getElementById("artificial");

// https://github.com/cazala/mnist

function colormap(x, min, max) {
    let val = Math.min(Math.max(x, min), max);
    let range = max - min;
    let red_v = Math.floor(255.0 * (val-min)/range);
    let green_v = Math.floor(127.5 * (val-min)/range);
    let blue_v = Math.floor(255.0 * (max-val)/range);
    return (red_v < 16 ? '#0' : '#') + red_v.toString(16)
        + (green_v < 16 ? '0' : '') + green_v.toString(16)
        + (blue_v < 16 ? '0' : '') + blue_v.toString(16);
}

function colormap2(x, min, mid, max) {
    if (x > mid)
        return colormap(x, mid, max);
    let val = Math.max(x, min);
    let range = mid - min;
    let blue_v = Math.floor(225.0 * (val-min)/range);
    return '#0000' + (blue_v < 16 ? '0' : '') + blue_v.toString(16);
}

const DRAW_RADIUS = 0.01;
const DRAW_GAP = 0.005;
const DRAW_AXON = 0.02;
const DRAW_LIGHTNING = 0.005;
class Neuron {
    constructor(arr, time, time2, x, y, func, radius=DRAW_RADIUS, drawAxon=true) {
        this.func = func; // activation function
        this.value = 0;
        this.synapses = []; // dendrites only
        this.bias = 0;
        this.time = time; // time for action potential to reach synapse
        this.conduction = time; // position of action potential along synapse
        this.diffusion = time2; // time for neurotransmitters to diffuse across synapse
        this.x = x;
        this.y = y;
        this.conducting = false; // used only for animation
        this.radius = radius; // used only for drawing
        this.drawAxon = drawAxon;
        arr.push(this);
    }

    connect(neuron, strength) {
        this.synapses.push({"neuron": neuron, "strength": strength});
    }

    draw(ctx, width, height) {
        ctx.fillStyle = colormap2(this.value, -1, 0, 1);
        ctx.lineWidth = 10.0/Math.sqrt(this.time);
        ctx.beginPath();
        ctx.arc(this.x * width, this.y * height, this.radius * width, 0, 2*Math.PI);
        ctx.fill();
        if (this.drawAxon) {
            for (var i = 0; i < this.synapses.length; i++) {
                let upstream = this.synapses[i].neuron;
                ctx.strokeStyle = colormap2(this.synapses[i].strength, -1, 0, 1);
                ctx.beginPath();
                ctx.moveTo((upstream.x + this.radius) * width, upstream.y * height);
                ctx.lineTo((upstream.x + this.radius + DRAW_AXON) * width, upstream.y * height);
                ctx.lineTo((this.x - this.radius - DRAW_GAP) * width, this.y * height);
                ctx.stroke();
            }
            if (this.conducting) {
                ctx.lineWidth = 2;
                ctx.strokeStyle = "#FFFF00";
                for (var i = 0; i < this.synapses.length; i++) {
                    let upstream = this.synapses[i].neuron;
                    var lightningX = upstream.x + this.radius;
                    var lightningY = upstream.y;
                    ctx.beginPath();
                    ctx.moveTo(lightningX * width, lightningY * height);
                    while (lightningX < this.x - this.radius - DRAW_GAP - Math.max((upstream.conduction)/(upstream.time)*(this.x - upstream.x - 2*this.radius - DRAW_GAP), 0)) {
                        lightningX += DRAW_LIGHTNING;
                        if (lightningX > upstream.x + this.radius + DRAW_AXON)
                            lightningY += (this.y - upstream.y)/(this.x - this.radius - DRAW_GAP - upstream.x)*DRAW_LIGHTNING;
                        ctx.lineTo(lightningX * width, (lightningY + 0.02*Math.random() - 0.01) * height);
                    }
                    ctx.stroke();
                
                }
            }
        }
    }

}

var art_neurons = [];
function sigmoid(x) {
    return 1.0/(1 + Math.exp(-x));
}
function sigmoid_derivative(y) {
    return y * (1-y);
    //return Math.exp(-x) * Math.pow(1 + Math.exp(-x), -2);
}

class ArtNeuron extends Neuron {
    constructor(x, y, radius, drawAxon) {
        super(art_neurons, 1, 0, x, y, sigmoid, radius, drawAxon);
    }

    turn() {
        let input = 0;
        for (var i = 0; i < this.synapses.length; i++) {
            let presynaptic = this.synapses[i].neuron;
            input += presynaptic.value * this.synapses[i].strength;
        }
        let output = this.func(input + this.bias);
        this.value = output;
        this.conduction = 0;
    }

    reset() {
        this.conduction = 1;
        this.value = 0;
    }
}

var bio_neurons = [];
function membrane_func(x) {
    let voltage = x - 70;
    let threshold = -55;
    if (voltage >= threshold)
        return 1;
    return 0;
}

class BioNeuron extends Neuron {
    constructor(x, y) {
        super(bio_neurons, 5, 3, x, y, membrane_func);
        // axon 1mm, 150 m/s for full myelination -> 6 µs conduction
        // 100 µs diffusion -> 3 ticks diffusion
        // max speed: 150 m/s (1 tick conduction)
        // min speed: 30 m/s (5 tick diffusion)
        // https://www.physiologyweb.com/calculators/diffusion_time_calculator.html
    }
}

var layers = [];
function makeLayer(num_neurons, get_x, get_y, previous, rad=DRAW_RADIUS, drawAxon=true) {
    var new_layer = [];
    for (var i = 0; i < num_neurons; i++) {
        let x = get_x(i);
        let y = get_y(i);
        let neuron = new ArtNeuron(x, y, rad, drawAxon);
        new_layer.push(neuron);
    }
    if (previous != null) {
        for (var i = 0; i < previous.length; i++) {
            for (var j = 0; j < new_layer.length; j++) {
                new_layer[j].connect(previous[i], Math.random() - 0.5);
            }
        }
    }
    layers.push(new_layer);
    return new_layer;
}
const LAYERS = 5;
var lastLayer = makeLayer(28*28,
    (i => 0.5/LAYERS + Math.floor(i/28)*0.01*5/8 - 0.05),
    (i => ((i%28)*0.01) + 0.36), null, 0.005);
lastLayer = makeLayer(20,
    (i => 1.5/LAYERS),
    (i => ((i%20)+0.5)/20), lastLayer, DRAW_RADIUS, false);
for (let j = 0; j < LAYERS-2; j++) {
    lastLayer = makeLayer(10,
        (i => (j+2.5)/LAYERS),
        (i => ((i%10)+0.5)/10), lastLayer, DRAW_RADIUS);
}

function runANNLayer(layer_num, delay, train=false) {
    console.log(layer_num);
    for (let j = 0; j < layers[layer_num].length; j++) {
        layers[layer_num][j].conducting = true;
        //layers[layer_num][j].turn();
        window.setTimeout(function() {
            layers[layer_num][j].turn();
            layers[layer_num][j].conducting = false;
        }, delay);
    }
    if (train) {
        window.setTimeout(function() {
            updateWeights(backPropagate([1,0,0,0,0,0,0,0,0,0]), 2.0);
        }, delay * 2);
        
    }
}

function runANN() {
    for (var i = 0; i < layers[0].length; i++) {
        layers[0][i].value = Math.random();
        layers[0][i].conduction = layers[0][i].time;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length; i++) {
        window.setTimeout(function() { runANNLayer(i, 2000, (i==layers.length-1)); }, (i-0.5) * 3000);
    }
}

// https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
function backPropagate(train_y) {
    var delta = {};
    for (var i = layers.length-1; i >= 1; i--) {
        let layer = layers[i];
        var errors = [];
        if (i == layers.length-1) {
            for (var j = 0; j < layer.length; j++)
                errors.push(layer[j].value - train_y[j])
        } else {
            for (var j = 0; j < layer.length; j++) {
                var error = 0;
                for (var k = 0; k < layers[i+1].length; k++)
                    error += layers[i+1][k].synapses[j].strength * delta[i+1][k];
                errors.push(error);
            }
        }
        var new_deltas = [];
        for (var j = 0; j < layer.length; j++) {
            new_deltas.push(errors[j] * sigmoid_derivative(layer[j].value))
        }
        delta[i] = new_deltas;
    }
    return delta;
}

function updateWeights(delta, l_rate) {
    for (var i = 1; i < layers.length; i++) {
        var inputs = [];
        for (var j = 0; j < layers[i-1].length; j++)
            inputs.push(layers[i-1][j].value);
        for (var j = 0; j < layers[i].length; j++) {
            for (var k = 0; k < inputs.length; k++)
                layers[i][j].synapses[k].strength -= l_rate * delta[i][j] * inputs[j];
            layers[i][j].bias -= l_rate * delta[i][j];
        }
    }
    console.log("Trained");
}

function drawDiagram() {
    var ctx = canvas1.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas1.width, canvas1.height);
    ctx = canvas2.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas2.width, canvas2.height);
    for (var i = 0; i < art_neurons.length; i++) {
        art_neurons[i].draw(ctx, canvas2.width, canvas2.height);
    }
}