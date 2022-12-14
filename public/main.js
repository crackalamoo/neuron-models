const SCALE = 4.0
var canvas1 = document.getElementById("biological");
var canvas2 = document.getElementById("artificial");

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
    constructor(arr, time, x, y, radius=DRAW_RADIUS, drawAxon=true) {
        this.value = 0;
        this.synapses = []; // dendrites
        this.bias = 0;
        this.time = time; // time for action potential to reach synapse
        this.conduction = time; // position of action potential along synapse
        this.x = x;
        this.y = y;
        this.conducting = false;
        this.radius = radius; // used only for drawing
        this.drawAxon = drawAxon;
        arr.push(this);
    }

}

function sigmoid(x, a, b) {
    return 1.0/(1 + Math.exp(-a*(x-b)));
}

function drawLightning(ctx, width, height, x0, y0, x1, y1, newLine=true) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#FFFF00";
    var x = x0;
    var y = y0;
    if (newLine) {
        ctx.beginPath();
        ctx.moveTo(x * width, y * height);
    }
    let distance = Math.sqrt(Math.pow(y1-y0,2) + Math.pow(x1-x0,2));
    let iterations = Math.floor(distance / DRAW_LIGHTNING);
    let iterX = (x1-x0)/distance;
    let iterY = (y1-y0)/distance;
    for (var i = 0; i < iterations; i++) {
        x += DRAW_LIGHTNING * iterX;
        y += DRAW_LIGHTNING * iterY;
        let offset = 0.02*Math.random() - 0.01;
        ctx.lineTo((x - offset*iterY) * width, (y + offset*iterX) * height);
    }
    ctx.stroke();
}

var ann_neurons = [];
function relu(x) {
    return Math.max(x, 0);
}
function relu_deriv(y) {
    if (y > 0)
        return 1;
    return 0;
}

class ANNNeuron extends Neuron {
    constructor(x, y, layer, index, radius, drawAxon) {
        super(ann_neurons, 1, x, y, radius, drawAxon);
        this.func = relu;
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

    draw(ctx, width, height) {
        ctx.fillStyle = colormap2(this.value, -1, 0, 1);
        ctx.lineWidth = 10.0;
        ctx.beginPath();
        ctx.arc(this.x * width, this.y * height, this.radius * width, 0, 2*Math.PI);
        ctx.fill();
        if (this.drawAxon) {
            for (var i = 0; i < this.synapses.length; i++) {
                let upstream = this.synapses[i].neuron;
                ctx.strokeStyle = colormap2(this.synapses[i].strength, -1, 0, 1);
                ctx.beginPath();
                ctx.moveTo((upstream.x + upstream.radius) * width, upstream.y * height);
                ctx.lineTo((upstream.x + upstream.radius + DRAW_AXON) * width, upstream.y * height);
                ctx.lineTo((this.x - this.radius - DRAW_GAP) * width, this.y * height);
                ctx.stroke();
            }
            if (this.conducting) {
                for (var i = 0; i < this.synapses.length; i++) {
                    let upstream = this.synapses[i].neuron;
                    drawLightning(ctx, width, height,
                        upstream.x + upstream.radius, upstream.y,  upstream.x + upstream.radius + DRAW_AXON, upstream.y);
                    drawLightning(ctx, width, height,
                        upstream.x + upstream.radius + DRAW_AXON, upstream.y,  this.x - this.radius - DRAW_GAP, this.y,  false);
                }
            }
        }
    }
}

var layers = [];
function makeLayer(num_neurons, get_x, get_y, previous, rad=DRAW_RADIUS, drawAxon=true) {
    var new_layer = [];
    for (var i = 0; i < num_neurons; i++) {
        let x = get_x(i);
        let y = get_y(i);
        let neuron = new ANNNeuron(x, y, new_layer, i, rad, drawAxon);
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
    (i => Math.floor(i/28)*0.01 + 0.36), null, 0.004);
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
var checkBioOutput = 0;

function loss_f(output, prediction) {
    return Math.power(prediction - output, 2);
}

function loss_f_deriv(output, prediction) {
    return 2 * (prediction - output);
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
        layers[0][i].conduction = 1;
    }
    for (var i = 1; i < layers.length; i++) {
        for (var j = 0; j < layers[i].length; j++)
            layers[i][j].reset();
    }
    for (let i = 1; i < layers.length; i++) {
        window.setTimeout(function() { runANNLayer(i, 1000); }, (i-0.5) * 1500);
    }
    window.setTimeout(function() {noANNControl = false;}, (layers.length+0.5) * 1500)
}

function runANN() {
    input = [];
    for (var i = 0; i < 28*28; i++)
        input.push(Math.random());
    runANNSample(input, [1,0,1,0,1,0,1,0,1,0]);
}

var bio_neurons = [];

// functions to determine how many channels to open

// voltage-gated ion channels of the soma
function naChannel(neuron) {
    let voltage = neuron.value;
    // Na channels must be allowed to close at peak voltage and not reopen until voltage is below -65 mV
    if (voltage < -65 || neuron.channels["na"][1] > 0) {
        let setPoint = sigmoid(voltage, 0.8, -50.5); // values chosen so that threshold will be near -55 mV
        if (neuron.value > 25)
            neuron.channels["na"][1] = 0;
        else
            neuron.channels["na"][1] = neuron.channels["na"][0] * setPoint;
    }
}

function kChannel(neuron) {
    let voltage = neuron.value;
    let setPoint = sigmoid(voltage, 0.3, 35);
    neuron.channels["k"][1] += 0.1 * (neuron.channels["k"][0] * setPoint - neuron.channels["k"][1]); // slow opening/closing compared to Na
}

// ligand-gated ion channels of the synapse
function ampa_open(synapse) {
    let glutamate = synapse.pre.glutamate * (1 - synapse.reuptake);
    let ampa_open_n =  synapse.receptors.ampa[0] * sigmoid(glutamate, 10, 0.55);
    synapse.receptors.ampa[1] = ampa_open_n;
}

function nmda_open(synapse) {
    let glutamate = synapse.pre.glutamate * (1 - synapse.reuptake);
    let mg_block = sigmoid(synapse.post.value, 0.3, -60);
    let nmda_open_n = synapse.receptors.nmda[0] * mg_block * sigmoid(glutamate, 10, 0.55);
    synapse.receptors.nmda[1] = nmda_open_n;
}

function gabaa_open(synapse) {
    let gaba = synapse.pre.gaba * (1 - synapse.reuptake);
    let gabaa_open_n = synapse.receptors.gabaa[0] * sigmoid(gaba, 10, 0.55);
    synapse.receptors.gabaa[1] = gabaa_open_n;
}

// neuroplasticity LTP/LTD calculator
function ltp_ltd(calcium) {
    return sigmoid(calcium, 1, 1) * Math.pow(calcium-0.5, 2)/(Math.pow(calcium-0.5, 2) + 1) - 0.1;
}

// extracellular ion concentrations
const EC_NA = 140; // mM
const EC_K = 5;
const EC_CL = 120;
const EC_CA = 2;
// intracellular ion concentrations
const IC_NA = 15;
const IC_K = 150;
const IC_CL = 5;
const IC_CA = 0;

const DIFFUSION_TIME = 3;
const REUPTAKE = 0.3;

class BioNeuron extends Neuron {
    constructor(x, y, glutamate=0, gaba=0, radius=DRAW_RADIUS, drawAxon=true, plastic=0, onFire=null) {
        super(bio_neurons, 5, x, y, radius, drawAxon);
        this.id = bio_neurons.length - 1;

        this.value = -70; // voltage

        this.forceDrawOn = false; // always draw this neuron with high voltage

        // relative permeability of each ion
        this.na = 2;
        this.k = 44;
        this.cl = 0;
        this.ca = 0;

        this.glutamate = glutamate;
        this.gaba = gaba;

        this.plastic = plastic; // neuroplasticity factor

        this.onFire = onFire; // optional function called when this neuron fires

        this.myelination = 0;

        this.presynaptic = {}; // map from presynaptic neuron ids to index of corresponding synapse
        this.postsynaptic = []; // list of postsynaptic neurons

        this.channels = {"na": [300,0], "k": [200,0]}; // relative number of total/open channels in soma, not synapses
    }

    connect(neuron, ampar, nmdar, gabaar, setTime=false, reuptake_speed=1) {
        this.presynaptic[neuron.id] = this.synapses.length;
        neuron.postsynaptic.push(this);
        this.synapses.push({"pre": neuron, "post": this, "reuptake": 1, "reuptake_speed": REUPTAKE * reuptake_speed,
            "receptors": {"ampa": [ampar,0], "nmda": [nmdar,0], "gabaa": [gabaar,0]}});
        if (setTime) {
            neuron.time = 150 * Math.sqrt(Math.pow(neuron.x - this.x, 2) + Math.pow(neuron.y - this.y, 2));
        }
    }

    reset() {
        this.conduction = this.time;
        this.conducting = false;
        this.value = -70;
        this.na = 2;
        this.k = 44;
        this.ca = 0;
        this.cl = 0;
        for (var i = 0; i < this.synapses.length; i++) {
            this.synapses[i].reuptake = 1;
        }
    }

    turn() {

        var setNa = 0;
        var setK = 0;
        var setCl = 0;
        var setCa = 0;

        // voltage-gated ion channels
        naChannel(this);
        setNa += this.channels["na"][1];

        kChannel(this);
        setK += this.channels["k"][1];

        // leak channels
        setNa += 2;
        setK += 48; // K is much more permeable than Na by default

        for (var i = 0; i < this.synapses.length; i++) {
            let synapse = this.synapses[i];

            // reuptake
            synapse.reuptake += synapse.reuptake_speed * (1 - synapse.reuptake);

            // ligand-gated ion channels
            ampa_open(synapse);
            setNa += synapse.receptors.ampa[1];
            setK += 0.3*synapse.receptors.ampa[1];

            nmda_open(synapse);
            setNa += synapse.receptors.nmda[1];
            setK += 0.3*synapse.receptors.nmda[1];
            setCa += synapse.receptors.nmda[1];

            gabaa_open(synapse);
            setCl += synapse.receptors.gabaa[1];

            // neuroplasticity
            if (this.plastic != 0 && plasticFactor != 0) {
                // Ca2+/NMDAR-based neuroplasticity of AMPAR
                let calcium = synapse.receptors.nmda[1];
                if (calcium > 0.1) {
                    let adj_ampa = ltp_ltd(calcium);
                    if (adj_ampa > 0)
                        synapse.receptors.ampa[0] += (15 - synapse.receptors.ampa[0]) * adj_ampa * this.plastic * plasticFactor;
                    else
                        synapse.receptors.ampa[0] -= (0 - synapse.receptors.ampa[0]) * adj_ampa * this.plastic * plasticFactor;
                }
            }
        }

        // allow channels to open and close based on set points
        this.na += 0.4 * (setNa - this.na);
        this.k += 0.08 * (setK - this.k);
        this.cl += 0.4 * (setCl - this.cl);
        this.ca += 0.08 * (setCa - this.ca);

        // membrane potential
        let top_sum = this.k*EC_K + this.na*EC_NA + this.cl*IC_CL + this.ca*Math.sqrt(EC_CA);
        let bottom_sum = this.k*IC_K + this.na*IC_NA + this.cl*EC_CL + this.ca*Math.sqrt(IC_CA);
        this.value = 61.0 * Math.log10(top_sum / bottom_sum);

        // action potential
        if (this.value > 0 && !this.conducting && this.channels["na"][1] > 0) {
            this.conducting = true;
            if (this.onFire != null) {
                this.onFire(this);
            }
            bioTimer = BIO_TIMER;
        }
        if (this.conducting) {
            if (this.conduction < this.time) {
                this.conduction = Math.min(this.time, this.conduction + this.myelination+1); // saltatory conduction
            } else {
                this.conduction++;
            }
            
            if (this.conduction >= this.time + DIFFUSION_TIME + 1) {
                // send neurotransmitters
                for (var i = 0; i < this.postsynaptic.length; i++) {
                    let post = this.postsynaptic[i];
                    let index = post.presynaptic[this.id];
                    post.synapses[index].reuptake *= 0.1;
                }

                // terminate action potential
                this.conducting = false;
                this.conduction = 0;
            }
        }
    }

    draw(ctx, width, height) {
        if (this.forceDrawOn)
            ctx.fillStyle = colormap(1, 0, 1);
        else
            ctx.fillStyle = colormap2(this.value, -80, -68, 0);
        ctx.beginPath();
        ctx.arc(this.x * width, this.y * height, this.radius * width, 0, 2*Math.PI);
        ctx.fill();

        if (this.drawAxon && this.postsynaptic.length > 0) {
            let drawTo = this.postsynaptic[0];
            let distance = Math.sqrt(Math.pow(this.x - drawTo.x, 2) + Math.pow(this.y - drawTo.y, 2));
            let drawCos = (drawTo.x - this.x)/distance;
            let drawSin = (drawTo.y - this.y)/distance;
            ctx.strokeStyle = colormap2(this.glutamate - this.gaba, -1.2, 0, 1.2);
            ctx.lineWidth = 0.6 * Math.sqrt(this.myelination + 1);
            ctx.beginPath();
            ctx.moveTo((this.x + this.radius*drawCos) * width, (this.y + this.radius*drawSin) * height);
            let endAxonX = drawTo.x - drawTo.radius * (drawCos*3);
            let endAxonY = drawTo.y - drawTo.radius * (drawSin*3);
            ctx.lineTo(endAxonX * width, endAxonY * height);
            ctx.stroke();
            ctx.lineWidth = 0.6;
            for (var i = 0; i < this.postsynaptic.length; i++) {
                let post = this.postsynaptic[i];
                let distance = Math.sqrt(Math.pow(endAxonX - post.x, 2) + Math.pow(endAxonY - post.y, 2));
                let drawCos = (post.x - endAxonX)/distance;
                let drawSin = (post.y - endAxonY)/distance;
                ctx.beginPath();
                ctx.moveTo(endAxonX * width, endAxonY * height);
                ctx.lineTo((post.x - (post.radius + DRAW_GAP) * drawCos) * width, (post.y - (post.radius + DRAW_GAP) * drawSin) * height);
                ctx.stroke();
            }
            if (this.conducting) {
                let conduct_pct = Math.min(1.0, this.conduction / this.time);
                drawLightning(ctx, width, height,
                    this.x + this.radius*drawCos, this.y + this.radius*drawSin,
                    (this.x + this.radius*drawCos)*(1-conduct_pct) + endAxonX*conduct_pct,
                    (this.y + this.radius*drawSin)*(1-conduct_pct) + endAxonY*conduct_pct);
            }
        }
    }
}

var rods = [];
for (var i = 0; i < 28*28; i++) {
    let neuron = new BioNeuron(0.5/LAYERS + (i%28)*0.01*5/8 - 0.05, Math.floor(i/28)*0.01 + 0.36, 1, 0, 0.004, false);
    neuron.time = 10;
    neuron.myelination = 100;
    rods.push(neuron);
}

function find_rod(x, y) {
    return rods[y*28 + x];
}


var visual_cortex_1 = [];

const PINWHEEL = [
    [[0,0],[-1,0],[1,0]], // shape -
    [[0,0],[0,-1],[0,1]], // shape |
    [[0,0],[1,1],[-1,-1]], // shape \
    [[0,0],[-1,1],[1,-1]] // shape /
];

for (var p = 0; p < PINWHEEL.length; p++) {
    // pinwheel orientation columns
    visual_cortex_1.push([]);
    let conv = PINWHEEL[p];
    for (var x = 2; x < 24; x++) {
        for (var y = 2; y < 24; y++) {
            let glutamate = Math.cbrt((Math.random() - 0.5)/4.0) + 0.5;
            let neuron = new BioNeuron(0.25 + 0.04*p + 0.02*Math.random(), 0.2 + 0.7*Math.random(), glutamate, 1.0 - glutamate, 0.005);
            neuron.connect(find_rod(x+1+conv[0][0], y+1+conv[0][1]), 5, 0, 0);
            neuron.connect(find_rod(x+1+conv[1][0], y+1+conv[1][1]), 1, 3, 0);
            neuron.connect(find_rod(x+1+conv[2][0], y+1+conv[2][1]), 1, 3, 0);
            neuron.myelination = 10;
            visual_cortex_1[p].push(neuron);
        }
    }
}


var p_temporal = [];
for (var i = 0; i < 120; i++) {
    let glutamate = Math.cbrt((Math.random() - 0.5)/4.0) + 0.5;
    let neuron = new BioNeuron(0.4 + 0.2*Math.random(), 0.5 + 0.4*Math.random(), glutamate, 1.0 - glutamate, 0.005);
    neuron.channels["na"][0] *= 0.88;
    let connectivity = 10 + Math.floor(10*Math.random());
    for (var j = 0; j < connectivity; j++) {
        let connect_to = visual_cortex_1[Math.floor(PINWHEEL.length * Math.random())][Math.floor(22*22*Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = visual_cortex_1[Math.floor(PINWHEEL.length * Math.random())][Math.floor(22*22*Math.random())];
        }
        neuron.connect(connect_to, (70.0+50*Math.random())*connect_to.glutamate / connectivity, 1,
            70.0*connect_to.gaba / connectivity + Math.random(), true);
    }
    neuron.myelination = 8 + Math.floor(3*Math.random());
    p_temporal.push(neuron);
}

for (var i = 0; i < 60; i++) {
    let neuron = p_temporal[Math.floor(p_temporal.length * Math.random())];
    for (var j = 0; j < 5; j++) {
        let connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1 || connect_to.id == neuron.id || neuron.postsynaptic.indexOf(connect_to) != -1) {
            connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        }
        neuron.connect(connect_to, 30.0*connect_to.glutamate, 2, 55.0*connect_to.gaba, true, Math.sqrt(connect_to.glutamate+0.4));
    }
}

const TEMPORAL_PLASTICITY = 0.4;

var a_temporal = [];
for (var i = 0; i < 50; i++) {
    let glutamate = Math.cbrt((Math.random() - 0.5)/4.0) + 0.5;
    let neuron = new BioNeuron(0.4 + 0.2*Math.random(), 0.2 + 0.29*Math.random(), glutamate, 1.0 - glutamate, 0.005, true, TEMPORAL_PLASTICITY);
    let connectivity = 7 + Math.floor(7*Math.random());
    for (var j = 0; j < connectivity; j++) {
        let connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        }
        neuron.connect(connect_to, (2+4*Math.random())*connect_to.glutamate, 2*connect_to.glutamate, 3*connect_to.gaba, true);
    }
    neuron.myelination = 10;
    a_temporal.push(neuron);
}

var wernicke = [];
for (var i = 0; i < 10; i++) {
    let neuron = new BioNeuron(0.74 + 0.02*Math.random(), 0.2+i*0.075, 1, 0, 0.005);
    neuron.myelination = 5;
    wernicke.push(neuron);
}

var bioOutput = -1;
function brocaFire(neuron) {
    if (bioOutput == -1) {
        bioOutput = broca.indexOf(neuron);
    }
}
const BROCA_PLASTICITY = 0.5;

var broca = [];
for (var i = 0; i < 10; i++) {
    let neuron = new BioNeuron(0.64 + 0.02*Math.random(), 0.2+i*0.075, 1, 0, 0.005, true, BROCA_PLASTICITY, brocaFire);
    for (var j = 0; j < 12; j++) {
        let connect_to = a_temporal[Math.floor(a_temporal.length * Math.random())];
        while (connect_to.postsynaptic.length > 6) {
            connect_to = a_temporal[Math.floor(a_temporal.length * Math.random())];
        }
        neuron.connect(connect_to, 0, 2, 0, true);
    }
    for (var j = 0; j < 30; j++) {
        let connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        while (connect_to.postsynaptic.indexOf(neuron) != -1) {
            connect_to = p_temporal[Math.floor(p_temporal.length * Math.random())];
        }
        neuron.connect(connect_to, 0, 2, 0);
    }
    neuron.connect(wernicke[i], 1.2, 0, 0, true, 0.01);
    neuron.myelination = 35;
    broca.push(neuron);
}

let stopSpeech = new BioNeuron(0.7, 0.51, 0, 1, 0.005);
stopSpeech.myelination = 25;
stopSpeech.channels["k"][0] *= 1.1;

broca[4].connect(stopSpeech, 0, 0, 25, true, 0.03);
for (var i = 0; i < 10; i++) {
    stopSpeech.connect(broca[i], 10, 5, 0, true);
    if (i != 4)
        broca[i].connect(stopSpeech, 0, 0, 25, false, 0.03);
}

let speechFeedback = new BioNeuron(0.73, 0.67, 1, 1, 0.005);
let speechFeedback2 = new BioNeuron(0.69, 0.60, 1, 0, 0.005);
let speechFeedback3 = new BioNeuron(0.72, 0.59, 1, 0, 0.005);
let speechFeedback4 = new BioNeuron(0.69, 0.75, 1, 0, 0.005);
speechFeedback2.connect(speechFeedback, 15, 5, 0, true);
speechFeedback3.connect(speechFeedback2, 15, 5, 0, true);
speechFeedback4.connect(speechFeedback3, 15, 5, 0, true);
stopSpeech.connect(speechFeedback4, 15, 5, 0);
speechFeedback.myelination = 1;
speechFeedback2.myelination = 0;
speechFeedback3.myelination = 0;
speechFeedback4.myelination = 0;

for (var i = 0; i < 10; i++) {
    speechFeedback.connect(broca[i], 10, 5, 0);
    broca[i].connect(speechFeedback, 0, 0, 15);
}
for (var i = 0; i < p_temporal.length; i += 2) {
    p_temporal[i].connect(speechFeedback, 0, 0, 30, false, 0.03);
    p_temporal[i].connect(stopSpeech, 0, 0, 20, false, 0.03);
}
for (var i = 0; i < a_temporal.length; i += 2) {
    a_temporal[i].connect(speechFeedback, 0, 0, 30, false, 0.03);
    a_temporal[i].connect(stopSpeech, 0, 0, 20, false, 0.03);
}



const DEBUG_TURNS = 25;
var debug_turns = 0;
function run_bioNeurons() {
    for (var i = 0; i < bio_neurons.length; i++) {
        bio_neurons[i].turn();
    }
    if (debug_turns > 0) {
        console.log(bio_neurons[0].ca);
        debug_turns--;
    }
    bioTimer--;
    if (bioTimer <= 0) {
        if (bioOutput == -1)
            bioOutput = -2;
        endBioInterval();
    }
}

function setVal(val) {
    bio_neurons[0].value = val;
    debug_turns = DEBUG_TURNS;
}

const BIO_TIMER = 30;
var bioTimer = BIO_TIMER;
function runBioSample(input, train=true) {
    bioOutput = -1;
    bioTimer = BIO_TIMER;
    for (var i = 0; i < bio_neurons.length; i++) {
        bio_neurons[i].reset();
    }
    startBioInterval();
    setTimeout(function() {
        for (var i = 0; i < rods.length; i++) {
            if (input[i] > 0.5) {
                rods[i].value = -50;
                rods[i].forceDrawOn = true;
            } else {
                rods[i].forceDrawOn = false;
            }
        }
    }, 300);
    if (train) {
        trainingBio = true;
        setTimeout(function() {
            wernicke[checkBioOutput].value = -50;
        }, 400);
    }
}

var bioInterval;
var runningBio = false;
function startBioInterval() {
    clearInterval(bioInterval);
    bioInterval = setInterval(run_bioNeurons, 150);
    runningBio = true;
}
function endBioInterval() {
    clearInterval(bioInterval);
    runningBio = false;
}


function drawAnnDiagram() {
    var ctx = canvas2.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas2.width, canvas2.height);
    for (var i = 0; i < ann_neurons.length; i++) {
        ann_neurons[i].draw(ctx, canvas2.width, canvas2.height);
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
            ctx.strokeStyle = "#00DD00";
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

function drawBioDiagram() {
    var ctx = canvas1.getContext('2d');
    ctx.fillStyle = "#444444";
    ctx.fillRect(0, 0, canvas1.width, canvas1.height);
    for (var i = 0; i < bio_neurons.length; i++) {
        bio_neurons[i].draw(ctx, canvas1.width, canvas1.height);
    }
    if (bioOutput != -1) {
        ctx.lineWidth = 10;
        let circleCell = bioOutput;
        ctx.strokeStyle = "#FF0000";
        if (bioOutput == -2) {
            circleCell = checkBioOutput;
            ctx.strokeStyle = "#FF5500";
        } else if (bioOutput == checkBioOutput) {
            ctx.strokeStyle = "#00DD00";
        }
        ctx.beginPath();
        ctx.arc(broca[circleCell].x * canvas1.width, broca[circleCell].y * canvas1.height, canvas1.width * 0.015, 0, 2*Math.PI);
        ctx.stroke();
    }
}