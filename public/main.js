const SCALE = 4.0
var canvas1 = document.getElementById("biological");
var canvas2 = document.getElementById("artificial");

// https://github.com/cazala/mnist
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4811948/
// The Brain Learns in Unexpected Ways - Scientific American
// https://www.nature.com/articles/nrn2575
// https://www.ncbi.nlm.nih.gov/books/NBK2532/

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
        this.synapses = []; // dendrites for ANN, axons for biological
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
function relu(x) {
    //return 1.0/(1 + Math.exp(-x));
    return Math.max(x, 0);
}
function relu_deriv(y) {
    if (y > 0)
        return 1;
    return 0;
}

class ArtNeuron extends Neuron {
    constructor(x, y, layer, index, radius, drawAxon) {
        super(art_neurons, 1, 0, x, y, relu, radius, drawAxon);
        this.layer = layer;
        this.index = index;
        this.deriv = relu_deriv;
    }

    connect(neuron, strength) {
        this.synapses.push({"neuron": neuron, "strength": strength});
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

const EC_NA = 140; // mM
const EC_K = 5;
const EC_CL = 120;
const EC_CA = 2;

class BioNeuron extends Neuron {
    constructor(x, y) {
        super(bio_neurons, 5, 3, x, y, membrane_func);
        // axon 1mm, 150 m/s for full myelination -> 6 µs conduction
        // 100 µs diffusion -> 3 ticks diffusion
        // max speed: 150 m/s (1 tick conduction)
        // min speed: 30 m/s (5 tick diffusion)
        // https://www.physiologyweb.com/calculators/diffusion_time_calculator.html

        this.value = -70;
        this.na = 15;
        this.k = 150;
        this.cl = 5;
        this.ca = 0;

        this.glutamate = 0;
        this.gaba = 0;

        this.receptors = []; // is this the best model? should this be synapse-specific?
    }

    connect(neuron) {
        this.synapses.push({"neuron": neuron, "calcium": 0, "reuptake": 1});
    }
}

var layers = [];
function makeLayer(num_neurons, get_x, get_y, previous, rad=DRAW_RADIUS, drawAxon=true) {
    var new_layer = [];
    for (var i = 0; i < num_neurons; i++) {
        let x = get_x(i);
        let y = get_y(i);
        let neuron = new ArtNeuron(x, y, new_layer, i, rad, drawAxon);
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
    (i => 0.5/LAYERS + (i%28)*0.01*5/8 - 0.05),
    (i => Math.floor(i/28)*0.01 + 0.36), null, 0.005);
lastLayer = makeLayer(20,
    (i => 1.5/LAYERS),
    (i => ((i%20)+0.5)/20), lastLayer, DRAW_RADIUS, false);
lastLayer = makeLayer(16,
    (i => 2.5/LAYERS),
    (i => ((i%16)+0.5)/16), lastLayer, DRAW_RADIUS);
for (let j = 0; j < LAYERS-3; j++) {
    lastLayer = makeLayer(10,
        (i => (j+3.5)/LAYERS),
        (i => ((i%10)+0.5)/10), lastLayer, DRAW_RADIUS);
}

function softmaxLastLayer() {
    var sum = 0;
    for (let i = 0; i < layers[layers.length-1].length; i++) {
        sum += layers[layers.length-1][i].value;
    }
    for (let i = 0; i < layers[layers.length-1].length; i++) {
        layers[layers.length-1][i].value /= sum;
    }
}
for (var i = 0; i < layers[layers.length-1].length; i++) {
    layers[layers.length-1][i].func = (x => Math.exp(x)); // exponential for softmax
    layers[layers.length-1][i].deriv = (x => x);
}

function runANNLayer(layer_num, delay) {
    console.log(layer_num);
    for (let j = 0; j < layers[layer_num].length; j++) {
        layers[layer_num][j].conducting = true;
        window.setTimeout(function() {
            layers[layer_num][j].turn();
            layers[layer_num][j].conducting = false;
        }, delay);
    }
    if (layer_num == layers.length - 1) {
        window.setTimeout(function() {
            softmaxLastLayer();
        }, delay+1);
    }
}

function ANNPredict(input) {
    for (var i = 0; i < layers[0].length; i++) {
        layers[0][i].value = input[i];
        layers[0][i].conduction = 1;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length-1; i++) {
        for (let j = 0; j < layers[i].length; j++)
            layers[i][j].turn();
    }
    for (let j = 0; j < layers[layers.length - 1].length; j++)
        layers[layers.length-1][j].turn();
    softmaxLastLayer();
    var output = [];
    for (let j = 0; j < layers[layers.length-1].length; j++)
        output.push(layers[layers.length-1][j].value);
    return output;
}

const LEARNING_RATE = 0.01;
var checkANNOutput = 0;

function loss_f(output, prediction) {
    return Math.power(prediction - output, 2);
}

function loss_f_deriv(output, prediction) {
    return 2 * (prediction - output);
}

// multiply two matrices stored as arrays
function matrix_multiply(a, b) {
    var result = [];
    for (var i = 0; i < a.length; i++) {
        result.push([]);
        for (var j = 0; j < b[0].length; j++) {
            result[i].push(0);
            for (var k = 0; k < a[0].length; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

function backPropagate(output, prediction) {
    var grad = []; // derivative of loss function with respect to activation of each neuron (not weights and biases)
    for (var i = 0; i < layers.length; i++) {
        grad.push([]);
        for (var j = 0; j < layers[i].length; j++) {
            grad[i].push(0);
        }
    }

    for (var i = layers.length - 1; i > 0; i--) {
        for (var j = 0; j < layers[i].length; j++) {
            if (i == layers.length - 1) {
                grad[i][j] = loss_f_deriv(output[j], prediction[j]);
            } else {
                var sum = 0;
                for (var k = 0; k < layers[i+1].length; k++) {
                    sum += grad[i+1][k] * layers[i+1][k].synapses[j].strength;
                }
                grad[i][j] = sum * layers[i][j].deriv(layers[i][j].value);
            }
        }
    }

    return grad;
}

function trainMiniBatch(batch) {
    for (var b = 0; b < batch.length; b++) {
        var input = batch[b][0];
        var output = batch[b][1];
        var prediction = ANNPredict(input);
        var grad = backPropagate(output, prediction);
        for (var i = 1; i < layers.length; i++) {
            for (var j = 0; j < layers[i].length; j++) {
                layers[i][j].bias -= LEARNING_RATE * grad[i][j];
                for (var k = 0; k < layers[i][j].synapses.length; k++) {
                    layers[i][j].synapses[k].strength -= LEARNING_RATE * grad[i][j] * layers[i-1][k].deriv(layers[i-1][k].value);
                }
            }
        }
    }
}

function digitOutput(digit) {
    var output = [0,0,0,0,0,0,0,0,0,0];
    output[digit] = 1;
    return output;
}

function runANNSample(input) {
    for (var i = 0; i < layers[0].length; i++) {
        layers[0][i].value = input[i];
        layers[0][i].conduction = layers[0][i].time;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length; i++) {
        window.setTimeout(function() { runANNLayer(i, 1000); }, (i-0.5) * 1500);
    }
}



function runANN() {
    input = [];
    for (var i = 0; i < 28*28; i++)
        input.push(Math.random());
    runANNSample(input, [1,0,1,0,1,0,1,0,1,0]);
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
    var maxDigit = 0;
    var outputSum = 0;
    for (var i = 0; i < layers[layers.length-1].length; i++) {
        outputSum += layers[layers.length-1][i].value;
        if (layers[layers.length-1][i].value > layers[layers.length-1][maxDigit].value)
            maxDigit = i;
    }
    if (outputSum > 0) {
        ctx.strokeStyle = "#FF0000";
        if (maxDigit == checkANNOutput)
            ctx.strokeStyle = "#00FF00";
        ctx.beginPath();
        ctx.arc(0.9 * canvas2.width, (0.05 + maxDigit * 0.1) * canvas2.height, canvas2.width*0.02, 0, 2*Math.PI);
        ctx.stroke();
    }
    ctx.font = ''+Math.round(canvas2.width*0.02)+"px Arial";
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    ctx.fillStyle = "#FFFFFF";
    for (var i = 0; i < 10; i++) {
        ctx.fillText(i, 0.9 * canvas2.width, (0.05 + i * 0.1) * canvas2.height);
    }
}