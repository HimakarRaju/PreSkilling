// script.js

// Set dimensions for the donut chart
const width = 400;
const height = 400;
const margin = 40;

// Create an SVG container for the chart
const svg = d3.select("#chart")
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", `translate(${width / 2}, ${height / 2})`);

// Create a color scale for the donut chart
const color = d3.scaleOrdinal(d3.schemeCategory10);

// Create an arc generator for the donut chart
const arc = d3.arc()
    .innerRadius(70) // Inner radius for the donut hole
    .outerRadius(100); // Outer radius for the donut

// Create a pie generator for calculating angles
const pie = d3.pie()
    .value(d => d.value) // Use the 'value' property for angles

// Initial dataset
let data = [
    { name: "Apples", value: 30 },
    { name: "Bananas", value: 20 },
    { name: "Cherries", value: 15 },
    { name: "Dates", value: 10 },
    { name: "Elderberries", value: 25 },
];

// Function to update the donut chart
function updateChart() {
    // Bind data to pie slices
    const arcs = svg.selectAll(".arc")
        .data(pie(data), d => d.data.name);

    // Remove old arcs
    arcs.exit()
        .transition()
        .duration(500)
        .attrTween("d", function(d) {
            const interpolate = d3.interpolate(this._current || d, d);
            this._current = interpolate(1); // Store the current arc
            return t => arc(interpolate(t)); // Return interpolated arc path
        })
        .remove();

    // Update existing arcs
    arcs.transition()
        .duration(500)
        .attrTween("d", function(d) {
            const interpolate = d3.interpolate(this._current || d, d);
            this._current = interpolate(1);
            return t => arc(interpolate(t));
        });

    // Add new arcs
    const newArcs = arcs.enter()
        .append("g")
        .attr("class", "arc");

    newArcs.append("path")
        .attr("d", arc)
        .attr("fill", d => color(d.data.name))
        .attr("opacity", 0)
        .transition()
        .duration(500)
        .attr("opacity", 1); // Fade in new arcs

    newArcs.append("text")
        .attr("transform", d => `translate(${arc.centroid(d)})`)
        .attr("dy", ".35em")
        .text(d => d.data.name)
        .style("text-anchor", "middle")
        .style("fill", "#fff");
}

// Update data function
function updateData() {
    // Generate new random data
    data = [
        { name: "Apples", value: Math.floor(Math.random() * 100) },
        { name: "Bananas", value: Math.floor(Math.random() * 100) },
        { name: "Cherries", value: Math.floor(Math.random() * 100) },
        { name: "Dates", value: Math.floor(Math.random() * 100) },
        { name: "Elderberries", value: Math.floor(Math.random() * 100) },
    ];
    updateChart(); // Call updateChart to redraw the chart
}

// Initial render of the donut chart
updateChart();
