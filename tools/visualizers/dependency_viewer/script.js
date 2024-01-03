//
// Copyright © 2022-2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
var cmdStream;

var viewportTop = 0;
var viewportLeft = 0;
var viewportScale = 1.0;

var canvasAgentWidth = 150;
var canvasHeaderHeight = 40;
var canvasStripeHeight = 50;
var canvasStripeWidth = 80;

var selectedAgentId = 0;
var selectedStripeId = 0;

var mouseDownX;
var mouseDownY;

function onCanvasContainerResized(e) {
    var canvas = document.getElementById("canvas");

    canvas.width = e.contentRect.width;
    canvas.height = e.contentRect.height;

    redraw();
}

function onGoButtonClicked() {
    var cmdStreamText = document.getElementById("cmdStreamText").value;
    parser = new DOMParser();
    cmdStream = parser.parseFromString(cmdStreamText, "text/xml");

    var numAgents = parseInt(cmdStream.getElementsByTagName("NUM_AGENTS")[0].innerHTML);
    var x = 0;
    var maxNumStripes = 0;
    for (var a = 0; a < numAgents; ++a) {
        var agentXml = cmdStream.getElementsByTagName("AGENT")[a];
        var agentType = agentXml.firstElementChild.nodeName;
        var numStripes = parseInt(agentXml.getElementsByTagName("NUM_STRIPES_TOTAL")[0].innerHTML);

        x += canvasAgentWidth;
        maxNumStripes = Math.max(maxNumStripes, numStripes);
    }

    viewportTop = 0;
    viewportLeft = 0;
    viewportScale = 1.0;
    selectedAgentId = 0;
    selectedStripeId = 0;

    redraw();
    document.getElementById("canvas").focus();
}

function redraw() {
    var canvas = document.getElementById("canvas");
    var ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.textAlign = "center";
    ctx.font = (14 * viewportScale) + "px Arial";

    var viewportRight = viewportLeft + canvas.width / viewportScale;
    var viewportBottom = viewportTop + canvas.height / viewportScale;
    var numAgents = parseInt(cmdStream.getElementsByTagName("NUM_AGENTS")[0].innerHTML);
    for (var a = 0; a < numAgents; ++a) {
        var x = a * canvasAgentWidth;
        if (x + canvasAgentWidth < viewportLeft) continue;
        if (x > viewportRight) continue;

        var agentXml = cmdStream.getElementsByTagName("AGENT")[a];
        var agentType = agentXml.firstElementChild.nodeName;
        var numStripes = parseInt(agentXml.getElementsByTagName("NUM_STRIPES_TOTAL")[0].innerHTML);

        ctx.fillStyle = agentType == "IFM_STREAMER" ? "darkgreen" :
            agentType == "OFM_STREAMER" ? "darkred" :
                "black";
        ctx.fillText("[" + a + "] " + agentType,
            (x + 0.5 * canvasAgentWidth - viewportLeft) * viewportScale,
            (canvasHeaderHeight - 10 - viewportTop) * viewportScale);
        ctx.fillStyle = "black"

        if (viewportScale > 0.05) {
            for (var s = 0; s < numStripes; ++s) {
                var r = getStripeRect(a, s);
                if (r.bottom < viewportTop) continue;
                if (r.top > viewportBottom) continue;

                ctx.lineWidth = (a == selectedAgentId && s == selectedStripeId) ? 5 : 1;
                ctx.strokeStyle = "black"
                ctx.beginPath();
                ctx.rect((r.left - viewportLeft) * viewportScale, (r.top - viewportTop) * viewportScale,
                    (r.right - r.left) * viewportScale, (r.bottom - r.top) * viewportScale);
                ctx.stroke();
                ctx.lineWidth = 1;

                if (viewportScale > 0.1) { // Speed up by skipping text when zoomed out too far
                    ctx.textBaseline = "top";
                    ctx.fillText(s, (x + 0.5 * canvasAgentWidth - viewportLeft) * viewportScale, (r.top + 5 - viewportTop) * viewportScale);
                    ctx.textBaseline = "bottom";
                }
            }
        } else {
            // Speed up by skipping individual stripe boxes when zoomed out too far
            ctx.fillStyle = "black"
            ctx.beginPath();
            var r1 = getStripeRect(a, 0);
            var r2 = getStripeRect(a, numStripes - 1);

            ctx.rect((r1.left - viewportLeft) * viewportScale, (r1.top - viewportTop) * viewportScale,
                (r1.right - r1.left) * viewportScale, (r2.bottom - r1.top) * viewportScale);
            ctx.fill();
            ctx.lineWidth = 1;
        }
    }

    if (viewportScale > 0.05) { // Speed up by skipping arrows when zoomed out too far
        var numSrcStripes = parseInt(cmdStream.getElementsByTagName("AGENT")[selectedAgentId].getElementsByTagName("NUM_STRIPES_TOTAL")[0].innerHTML);
        var stripesToDraw = document.getElementById("forAllStripesCheck").checked ? [...Array(numSrcStripes).keys()] : [selectedStripeId];

        GetLargestNeededStripeId

        var agentXml = cmdStream.getElementsByTagName("AGENT")[selectedAgentId];
        var numDependencies = agentXml.getElementsByTagName("DEPENDENCY").length;
        var agentType = agentXml.firstElementChild.nodeName;
        for (var dependencyId = 0; dependencyId < numDependencies; ++dependencyId) {
            for (var srcStripeId of stripesToDraw) {
                var dep = parseDependency(selectedAgentId, "DEPENDENCY", dependencyId);
                var y = 0;
                if (dep.writesToTileSize >= 0)
                {
                    if (srcStripeId < dep.writesToTileSize) continue;
                    y = GetLastReaderOfEvictedStripeId(dep, srcStripeId, dep.writesToTileSize);
                }
                else
                {
                    y = GetLargestNeededStripeId(dep, srcStripeId);
                }
                if (dep.useForCmdStream && dep.useForScheduling)
                {
                    ctx.strokeStyle = "blue"
                }
                else if (dep.useForCmdStream)
                {
                    ctx.strokeStyle = "green"
                }
                else
                {
                    ctx.strokeStyle = "red"
                }
                drawArrowBetweenStripes(ctx, selectedAgentId, srcStripeId, dep.otherAgentId, y);
            }
        }
    }
}

function parseDependency(agentId, dependency_type, dependencyId) {
    var depXml = cmdStream.getElementsByTagName("AGENT")[agentId].getElementsByTagName(dependency_type)[dependencyId];
    var dep = {
        otherAgentId: parseInt(depXml.getElementsByTagName("OTHER_AGENT_ID")[0].innerHTML),
        outerRatio: {
            other: parseInt(depXml.getElementsByTagName("OUTER_RATIO")[0].getElementsByTagName("OTHER")[0].innerHTML),
            self: parseInt(depXml.getElementsByTagName("OUTER_RATIO")[0].getElementsByTagName("SELF")[0].innerHTML),
        },
        innerRatio: {
            other: parseInt(depXml.getElementsByTagName("INNER_RATIO")[0].getElementsByTagName("OTHER")[0].innerHTML),
            self: parseInt(depXml.getElementsByTagName("INNER_RATIO")[0].getElementsByTagName("SELF")[0].innerHTML),
        },
        boundary: parseInt(depXml.getElementsByTagName("BOUNDARY")[0].innerHTML),
        writesToTileSize: parseInt(depXml.getElementsByTagName("WRITES_TO_TILE_SIZE")[0].innerHTML),
        useForCmdStream: parseInt(depXml.getElementsByTagName("USE_FOR_COMMAND_STREAM")[0].innerHTML) == 1,
        useForScheduling: parseInt(depXml.getElementsByTagName("USE_FOR_SCHEDULING")[0].innerHTML) == 1,
    }
    return dep;
}

function getStripeRect(agentId, stripeId) {
    return {
        left: agentId * canvasAgentWidth + 0.5 * (canvasAgentWidth - canvasStripeWidth),
        top: stripeId * canvasStripeHeight + canvasHeaderHeight,
        right: (agentId + 1) * canvasAgentWidth - 0.5 * (canvasAgentWidth - canvasStripeWidth),
        bottom: stripeId * canvasStripeHeight + canvasStripeHeight + canvasHeaderHeight,
    }
}

function drawArrowBetweenStripes(ctx, fromAgentId, fromStripeId, toAgentId, toStripeId) {
    var fromRect = getStripeRect(fromAgentId, fromStripeId);
    var toRect = getStripeRect(toAgentId, toStripeId);
    drawArrow(ctx, (0.5 * (fromRect.left + fromRect.right) - viewportLeft) * viewportScale,
        (0.5 * (fromRect.top + fromRect.bottom) - viewportTop) * viewportScale,
        (0.5 * (toRect.left + toRect.right) - viewportLeft) * viewportScale,
        (0.5 * (toRect.top + toRect.bottom) - viewportTop) * viewportScale);
}

function drawArrow(ctx, ax, ay, bx, by) {
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);

    var l = Math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
    var vx = (bx - ax) / l;
    var vy = (by - ay) / l;
    var angle = 30 * Math.PI / 180;

    var rx = -Math.cos(angle) * vx - Math.sin(angle) * vy;
    var ry = Math.sin(angle) * vx - Math.cos(angle) * vy;
    ctx.lineTo(bx + rx * 20, by + ry * 20);

    var rx = -Math.cos(-angle) * vx - Math.sin(-angle) * vy;
    var ry = Math.sin(-angle) * vx - Math.cos(-angle) * vy;
    ctx.moveTo(bx, by);
    ctx.lineTo(bx + rx * 20, by + ry * 20);

    ctx.stroke();
}

function getStripeFromCanvasPos(x, y) {
    var agentId = Math.floor((x / viewportScale + viewportLeft) / canvasAgentWidth);
    var stripeId = Math.floor((y / viewportScale + viewportTop - canvasHeaderHeight) / canvasStripeHeight);
    return {
        agentId: agentId,
        stripeId: stripeId
    };
}

function onCanvasMouseDown(event) {
    if (event.button == 0) {
        mouseDownX = event.offsetX;
        mouseDownY = event.offsetY;
    }
}

function onCanvasMouseMove(event) {
    if (event.buttons == 1) {
        if (Math.abs(event.offsetX - mouseDownX) > 3 || Math.abs(event.offsetY - mouseDownY) > 3) {
            viewportLeft -= event.movementX / viewportScale;
            viewportTop -= event.movementY / viewportScale;
            redraw();
        }
    }
}

function onCanvasMouseUp(event) {
    if (event.button == 0) {
        if (Math.abs(event.offsetX - mouseDownX) < 3 && Math.abs(event.offsetY - mouseDownY) < 3) {
            var a = getStripeFromCanvasPos(event.offsetX, event.offsetY);
            selectedAgentId = a.agentId;
            selectedStripeId = a.stripeId;
            redraw();
        }
    }
}

function refreshStripeId() {
    var maximum = parseInt(cmdStream.getElementsByTagName("AGENT")[selectedAgentId].getElementsByTagName("NUM_STRIPES_TOTAL")[0].innerHTML) - 1;
    var minimum = 0;
    selectedStripeId = Math.min(maximum, Math.max(minimum, selectedStripeId))
}

function scrollToStripe(agentId, stripeId) {
    var r = getStripeRect(agentId, stripeId);
    var canvas = document.getElementById("canvas");
    if (r.left > (viewportLeft + (canvas.width / viewportScale))) {
        viewportLeft = r.right - canvas.width / viewportScale + 10;
    }
    if (r.right < viewportLeft) {
        viewportLeft = r.left - 10;
    }
    if (r.bottom < viewportTop) {
        viewportTop = r.top - 10;
    }
    if (r.top > (viewportTop + (canvas.height / viewportScale))) {
        viewportTop = r.bottom - canvas.height / viewportScale + 10;
    }
}

function onCanvasKeyDown(event) {
    if (event.code == "KeyW") {
        viewportTop -= 100 / viewportScale;
        redraw();
    } else if (event.code == "KeyS") {
        viewportTop += 100 / viewportScale;
        redraw();
    } else if (event.code == "KeyA") {
        viewportLeft -= 100 / viewportScale;
        redraw();
    } else if (event.code == "KeyD") {
        viewportLeft += 100 / viewportScale;
        redraw();
    } else if (event.code == "ArrowLeft") {
        selectedAgentId = Math.max(0, selectedAgentId - 1);
        refreshStripeId()
        scrollToStripe(selectedAgentId, selectedStripeId);
        redraw();
    } else if (event.code == "ArrowRight") {
        selectedAgentId = Math.min(cmdStream.getElementsByTagName("AGENT").length - 1, selectedAgentId + 1);
        refreshStripeId();
        scrollToStripe(selectedAgentId, selectedStripeId);
        redraw();
    } else if (event.code == "ArrowUp") {
        selectedStripeId = Math.max(0, selectedStripeId - 1);
        scrollToStripe(selectedAgentId, selectedStripeId);
        redraw();
    } else if (event.code == "ArrowDown") {
        var maximum = parseInt(cmdStream.getElementsByTagName("AGENT")[selectedAgentId].getElementsByTagName("NUM_STRIPES_TOTAL")[0].innerHTML) - 1;
        selectedStripeId = Math.min(maximum, selectedStripeId + 1);
        scrollToStripe(selectedAgentId, selectedStripeId);
        redraw();
    }
    else if (event.code == "KeyG") {
        var s = prompt("Agent ID/Stripe ID", selectedAgentId + "/" + selectedStripeId);
        if (s) {
            p = s.split("/");
            selectedAgentId = parseInt(p[0]);
            selectedStripeId = p.length > 1 ? parseInt(p[1]) : 0;
            scrollToStripe(selectedAgentId, selectedStripeId);
            redraw();
        }
    }
}

function onGlobalKeyDown(event) {
    if (event.code == "Enter" && event.getModifierState("Control")) {
        onGoButtonClicked();
    }
}


function onCanvasWheel(event) {
    // Prevent scrolling the background web page when using the scroll wheel on the canvas
    // If we didn't do this the screen would "wiggle" when zooming in and out.
    event.preventDefault();
    var worldX = event.offsetX / viewportScale + viewportLeft;
    var l = (worldX - viewportLeft) * viewportScale;
    var worldY = event.offsetY / viewportScale + viewportTop;
    var t = (worldY - viewportTop) * viewportScale;

    if (event.deltaY < 0) {
        viewportScale *= 1.1;
    } else {
        viewportScale /= 1.1;
    }

    viewportLeft = worldX - l / viewportScale;
    viewportTop = worldY - t / viewportScale;

    redraw();
}


document.getElementById("goButton").addEventListener("click", onGoButtonClicked);
document.getElementById("canvas").addEventListener("mousedown", onCanvasMouseDown);
document.getElementById("canvas").addEventListener("mousemove", onCanvasMouseMove);
document.getElementById("canvas").addEventListener("mouseup", onCanvasMouseUp);
document.getElementById("canvas").addEventListener("wheel", onCanvasWheel);
document.getElementById("canvas").addEventListener("keydown", onCanvasKeyDown);
document.getElementById("forAllStripesCheck").addEventListener("change", redraw);

window.addEventListener("keydown", onGlobalKeyDown);

var resizeObserver = new ResizeObserver(entries => { for (let e of entries) {
    if (e.target.id == "canvas-container") {
        onCanvasContainerResized(e);
    }
} });
resizeObserver.observe(document.getElementById("canvas-container"));

onGoButtonClicked();