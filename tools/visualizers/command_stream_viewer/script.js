//
// Copyright © 2023 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

var cmdStream;
var agents;
var waitForCounterToAgentAndCommandIdxWithinAgent;

var viewportTop = 0;
var viewportLeft = 0;
var viewportScale = 1.0;

var canvasAgentWidth = 280;
var canvasHeaderHeight = 40;
var canvasCommandHeight = 30;
var canvasCommandWidth = 250;

var selectedAgentId = 0;
var selectedCommandIdxWithinAgent = 0;

var mouseDownX;
var mouseDownY;

function onCanvasContainerResized(e) {
    var canvas = document.getElementById("canvas");

    canvas.width = e.contentRect.width;
    canvas.height = e.contentRect.height;

    redraw();
}

function parseCommandsArray(commandTypeIn)
{
    var dmaRdCounter = 0;
    var dmaWrCounter = 0;
    var mceifCounter = 0;
    var mceStripeCounter = 0;
    var pleCodeLoadedIntoPleSramCounter = 0;
    var pleStripeCounter = 0;

    var commandsXml = cmdStream.getElementsByTagName(commandTypeIn)[0];
    for (var commandIdx = 0; commandIdx < commandsXml.children.length; ++commandIdx) {
        var commandXml = commandsXml.children[commandIdx];
        commandXml.commandIdx = commandIdx;
        var commandType = commandXml.tagName;

        var agentId;
        // All command types apart from WaitForCounter have an agent ID, otherwise we have to infer it
        if (commandType == "WAIT_FOR_COUNTER_COMMAND")
        {
            // Look for the next command that isn't a WaitForCounter, to determine which agent to associate this with
            for (var lookaheadCommandIdx = commandIdx + 1; lookaheadCommandIdx < commandsXml.children.length; ++lookaheadCommandIdx) {
                var lookaheadCommandXml = commandsXml.children[lookaheadCommandIdx];
                var lookaheadCommandType = lookaheadCommandXml.tagName;
                if (lookaheadCommandType != "WAIT_FOR_COUNTER_COMMAND")
                {
                    agentId = parseInt(lookaheadCommandXml.getElementsByTagName("AGENT_ID")[0].innerHTML)
                    break;
                }
            }
        }
        else
        {
            agentId = parseInt(commandXml.getElementsByTagName("AGENT_ID")[0].innerHTML)
        }

        commandXml.commandIdxWithinAgent = agents[agentId].commands.length;
        commandXml.agentId = agentId;

        // Store the counter value that we would have after this command completes
        if (commandTypeIn == "DMA_RD_COMMANDS" && commandType == "DMA_COMMAND")
        {
            dmaRdCounter += 1;
            commandXml.DmaRdCounter = dmaRdCounter;
            commandXml.counterValue = dmaRdCounter;
        }
        else if (commandTypeIn == "DMA_WR_COMMANDS" && commandType == "DMA_COMMAND")
        {
            dmaWrCounter += 1;
            commandXml.DmaWrCounter = dmaWrCounter;
            commandXml.counterValue = dmaWrCounter;
        }
        else if (commandTypeIn == "MCE_COMMANDS" && commandType == "START_MCE_STRIPE_COMMAND")
        {
            mceStripeCounter += 1;
            commandXml.MceStripeCounter = mceStripeCounter;
            commandXml.counterValue = mceStripeCounter;
        }
        else if (commandTypeIn == "MCE_COMMANDS" && commandType == "CONFIG_MCEIF_COMMAND")
        {
            mceifCounter += 1;
            commandXml.MceifCounter = mceifCounter;
            commandXml.counterValue = mceifCounter;
        }
        else if (commandTypeIn == "PLE_COMMANDS" && commandType == "LOAD_PLE_CODE_INTO_PLE_SRAM_COMMAND")
        {
            pleCodeLoadedIntoPleSramCounter += 1;
            commandXml.PleCodeLoadedIntoPleSramCounter = pleCodeLoadedIntoPleSramCounter;
            commandXml.counterValue = pleCodeLoadedIntoPleSramCounter;
        }
        else if (commandTypeIn == "PLE_COMMANDS" && commandType == "START_PLE_STRIPE_COMMAND")
        {
            pleStripeCounter += 1;
            commandXml.PleStripeCounter = pleStripeCounter;
            commandXml.counterValue = pleStripeCounter;
        }

        agents[agentId].commands.push(commandXml);
    }
}

function onGoButtonClicked() {
    var cmdStreamText = document.getElementById("cmdStreamText").value;
    parser = new DOMParser();
    cmdStream = parser.parseFromString(cmdStreamText, "text/xml");

    agents = new Array();
    var agentsXml = cmdStream.getElementsByTagName("AGENTS")[0];
    for (var agentId = 0; agentId < agentsXml.children.length; ++agentId) {
        var agentXml = agentsXml.children[agentId];

        agents.push({ commands: new Array() });
    }
    waitForCounterToAgentAndCommandIdxWithinAgent = new Map();
    parseCommandsArray("DMA_RD_COMMANDS");
    parseCommandsArray("DMA_WR_COMMANDS");
    parseCommandsArray("MCE_COMMANDS");
    parseCommandsArray("PLE_COMMANDS");

    // Figure out where we should draw arrows for WaitForCounter commands
    for (var agentId = 0; agentId < agents.length; agentId++)
    {
        var numSrcCommands = parseInt(agents[agentId].commands.length);
        for (var commandIdx = 0; commandIdx < numSrcCommands; commandIdx++)
        {
            var commandXml = agents[agentId].commands[commandIdx];
            var commandType = commandXml.tagName;
            if (commandType == "WAIT_FOR_COUNTER_COMMAND")
            {
                var counterName = commandXml.getElementsByTagName("COUNTER_NAME")[0].innerHTML;
                var counterFieldName = counterName + "Counter"
                var counterValue = parseInt(commandXml.getElementsByTagName("COUNTER_VALUE")[0].innerHTML);
                // Find the right destination agent and right destination command
                for (var destAgentId = 0; destAgentId < agents.length; destAgentId++)
                {
                    for (var destCommandIdxWithinAgent = 0; destCommandIdxWithinAgent < agents[destAgentId].commands.length; destCommandIdxWithinAgent++)
                    {
                        if (agents[destAgentId].commands[destCommandIdxWithinAgent][counterFieldName] == counterValue)
                        {
                            waitForCounterToAgentAndCommandIdxWithinAgent.set(agentId*10000+commandIdx, {
                                destAgentId: destAgentId,
                                destCommandIdxWithinAgent: destCommandIdxWithinAgent
                            });
                        }
                    }
                }
            }
        }
    }

    viewportTop = 0;
    viewportLeft = 0;
    viewportScale = 1.0;
    selectedAgentId = 0;
    selectedCommandIdxWithinAgent = 0;

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
    var agentsXml = cmdStream.getElementsByTagName("AGENTS")[0];
    for (var agentId = 0; agentId < agentsXml.children.length; ++agentId) {
        var agent = agents[agentId];

        var x = agentId * canvasAgentWidth;
        if (x + canvasAgentWidth < viewportLeft) continue;
        if (x > viewportRight) continue;

        var agentXml = agentsXml.children[agentId];
        var agentType = agentXml.tagName;

        ctx.fillStyle = agentType == "IFM_STREAMER" ? "darkgreen" :
            agentType == "OFM_STREAMER" ? "darkred" :
                "black";
        ctx.fillText("[" + agentId + "] " + agentType,
            (x + 0.5 * canvasAgentWidth - viewportLeft) * viewportScale,
            (canvasHeaderHeight - 10 - viewportTop) * viewportScale);
        ctx.fillStyle = "black"

        if (viewportScale > 0.05) {
            for (var commandIdxWithinAgent = 0; commandIdxWithinAgent < agent.commands.length; ++commandIdxWithinAgent) {
                var commandXml = agent.commands[commandIdxWithinAgent];
                var commandType = commandXml.tagName;

                var r = getCommandRect(agentId, commandIdxWithinAgent);
                if (r.bottom < viewportTop) continue;
                if (r.top > viewportBottom) continue;

                ctx.lineWidth = (agentId == selectedAgentId && commandIdxWithinAgent == selectedCommandIdxWithinAgent) ? 5 : 1;
                ctx.strokeStyle = "black"
                ctx.beginPath();
                ctx.rect((r.left - viewportLeft) * viewportScale, (r.top - viewportTop) * viewportScale,
                    (r.right - r.left) * viewportScale, (r.bottom - r.top) * viewportScale);
                ctx.stroke();
                ctx.lineWidth = 1;

                if (viewportScale > 0.1) { // Speed up by skipping text when zoomed out too far
                    ctx.textBaseline = "top";

                    if (commandXml.parentElement.tagName == "DMA_RD_COMMANDS") {
                        ctx.fillStyle = "green"
                    } else if (commandXml.parentElement.tagName == "DMA_WR_COMMANDS") {
                        ctx.fillStyle = "blue"
                    } else if (commandXml.parentElement.tagName == "MCE_COMMANDS") {
                        ctx.fillStyle = "goldenrod"
                    } else if (commandXml.parentElement.tagName == "PLE_COMMANDS") {
                        ctx.fillStyle = "DeepPink"
                    }

                    var text = commandXml.commandIdx + ": ";
                    if (commandType == "WAIT_FOR_COUNTER_COMMAND") {
                        text = text + "WaitForCounter (" + commandXml.getElementsByTagName("COUNTER_NAME")[0].innerHTML + ", " + commandXml.getElementsByTagName("COUNTER_VALUE")[0].innerHTML + ")";
                    } else
                    {
                        var text = text + commandType.replace("_COMMAND", "");
                        if (commandXml.counterValue != undefined) {
                            text += " -> " + commandXml.counterValue;
                        }
                    }
                    ctx.fillText(text, (x + 0.5 * canvasAgentWidth - viewportLeft) * viewportScale, (r.top + 5 - viewportTop) * viewportScale);
                    ctx.textBaseline = "bottom";
                }
            }
        } else {
            // Speed up by skipping individual command boxes when zoomed out too far
            ctx.fillStyle = "black"
            ctx.beginPath();
            var r1 = getCommandRect(agentId, 0);
            var r2 = getCommandRect(agentId, agent.commands.length - 1);

            ctx.rect((r1.left - viewportLeft) * viewportScale, (r1.top - viewportTop) * viewportScale,
                (r1.right - r1.left) * viewportScale, (r2.bottom - r1.top) * viewportScale);
            ctx.fill();
            ctx.lineWidth = 1;
        }
    }

    if (viewportScale > 0.05) { // Speed up by skipping arrows when zoomed out too far
        var srcAgents = (document.getElementById("forAllCommandsInAgentSelect").value == "everything") ? [...Array(agents.length).keys()] : [selectedAgentId];
        for (var agentId of srcAgents) {
            var numSrcCommands = parseInt(agents[agentId].commands.length);
            var commandsToDraw = (document.getElementById("forAllCommandsInAgentSelect").value != "selectedCommand") ? [...Array(numSrcCommands).keys()] : [selectedCommandIdxWithinAgent];

            for (var srcCommandIdxWithinAgent of commandsToDraw) {
                var commandXml = agents[agentId].commands[srcCommandIdxWithinAgent];
                var commandType = commandXml.tagName;

                // Arrows to show dependencies between command queues
                if (document.getElementById("waitForCounterCheck").checked) {
                    if (commandType == "WAIT_FOR_COUNTER_COMMAND")
                    {
                        var agentAndCommandIdxWithinAgent = waitForCounterToAgentAndCommandIdxWithinAgent.get(agentId*10000+srcCommandIdxWithinAgent);
                        ctx.strokeStyle = "red";
                        drawArrowBetweenCommands(ctx, agentId, srcCommandIdxWithinAgent, agentAndCommandIdxWithinAgent.destAgentId, agentAndCommandIdxWithinAgent.destCommandIdxWithinAgent);
                    }
                }

                // Arrows to show the next command in the queue, if it's not the one immediately below it (i.e. from a different agent)
                if (document.getElementById("commandOrderCheck").checked) {
                    var nextCommandXml = commandXml.nextElementSibling;
                    if (nextCommandXml) {
                        if (nextCommandXml.agentId != agentId) {
                            if (commandXml.parentElement.tagName == "DMA_RD_COMMANDS") {
                                ctx.strokeStyle = "green"
                            } else if (commandXml.parentElement.tagName == "DMA_WR_COMMANDS") {
                                ctx.strokeStyle = "blue"
                            } else if (commandXml.parentElement.tagName == "MCE_COMMANDS") {
                                ctx.strokeStyle = "goldenrod"
                            } else if (commandXml.parentElement.tagName == "PLE_COMMANDS") {
                                ctx.strokeStyle = "DeepPink"
                            }

                            drawArrowBetweenCommands(ctx, agentId, srcCommandIdxWithinAgent, nextCommandXml.agentId, nextCommandXml.commandIdxWithinAgent);
                        }
                    }
                }
            }
        }
    }
}

function getCommandRect(agentId, commandId) {
    return {
        left: agentId * canvasAgentWidth + 0.5 * (canvasAgentWidth - canvasCommandWidth),
        top: commandId * canvasCommandHeight + canvasHeaderHeight,
        right: (agentId + 1) * canvasAgentWidth - 0.5 * (canvasAgentWidth - canvasCommandWidth),
        bottom: commandId * canvasCommandHeight + canvasCommandHeight + canvasHeaderHeight,
    }
}

function drawArrowBetweenCommands(ctx, fromAgentId, fromCommandIdxWithinAgent, toAgentId, toCommandIdxWithinAgent) {
    var fromRect = getCommandRect(fromAgentId, fromCommandIdxWithinAgent);
    var toRect = getCommandRect(toAgentId, toCommandIdxWithinAgent);

    var xFraction = fromAgentId < toAgentId ? 1 : 0;

    var fromX = (fromRect.left + xFraction * (fromRect.right - fromRect.left) - viewportLeft) * viewportScale;
    var fromY = (0.5 * (fromRect.top + fromRect.bottom) - viewportTop) * viewportScale;

    var toX = (toRect.left + (1 - xFraction) * (toRect.right - toRect.left) - viewportLeft) * viewportScale;
    var toY = (0.5 * (toRect.top + toRect.bottom) - viewportTop) * viewportScale;

    drawArrow(ctx, fromX, fromY, toX, toY);
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

function getCommandFromCanvasPos(x, y) {
    var agentId = Math.floor((x / viewportScale + viewportLeft) / canvasAgentWidth);
    var commandId = Math.floor((y / viewportScale + viewportTop - canvasHeaderHeight) / canvasCommandHeight);
    return {
        agentId: agentId,
        commandId: commandId
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
            var a = getCommandFromCanvasPos(event.offsetX, event.offsetY);
            selectedAgentId = a.agentId;
            selectedCommandIdxWithinAgent = a.commandId;

            redraw();
        }
    }
}

function validateSelectedCommandIdxWithinAgent() {
    var maximum = agents[selectedAgentId].commands.length - 1;
    var minimum = 0;
    selectedCommandIdxWithinAgent = Math.min(maximum, Math.max(minimum, selectedCommandIdxWithinAgent))
}

function scrollToCommand(agentId, commandId) {
    var r = getCommandRect(agentId, commandId);
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
        validateSelectedCommandIdxWithinAgent()
        scrollToCommand(selectedAgentId, selectedCommandIdxWithinAgent);
        redraw();
    } else if (event.code == "ArrowRight") {
        selectedAgentId = Math.min(cmdStream.getElementsByTagName("AGENTS")[0].children.length - 1, selectedAgentId + 1);
        validateSelectedCommandIdxWithinAgent();
        scrollToCommand(selectedAgentId, selectedCommandIdxWithinAgent);
        redraw();
    } else if (event.code == "ArrowUp") {
        selectedCommandIdxWithinAgent = Math.max(0, selectedCommandIdxWithinAgent - 1);
        scrollToCommand(selectedAgentId, selectedCommandIdxWithinAgent);
        redraw();
    } else if (event.code == "ArrowDown") {
        var maximum = agents[selectedAgentId].commands.length - 1;
        selectedCommandIdxWithinAgent = Math.min(maximum, selectedCommandIdxWithinAgent + 1);
        scrollToCommand(selectedAgentId, selectedCommandIdxWithinAgent);
        redraw();
    }
    else if (event.code == "KeyG") {
        var s = prompt("Agent ID/Command Idx within agent", selectedAgentId + "/" + selectedCommandIdxWithinAgent);
        if (s) {
            p = s.split("/");
            selectedAgentId = parseInt(p[0]);
            selectedCommandIdxWithinAgent = p.length > 1 ? parseInt(p[1]) : 0;
            scrollToCommand(selectedAgentId, selectedCommandIdxWithinAgent);
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
document.getElementById("forAllCommandsInAgentSelect").addEventListener("change", redraw);
document.getElementById("waitForCounterCheck").addEventListener("change", redraw);
document.getElementById("commandOrderCheck").addEventListener("change", redraw);


window.addEventListener("keydown", onGlobalKeyDown);

var resizeObserver = new ResizeObserver(entries => { for (let e of entries) {
    if (e.target.id == "canvas-container") {
        onCanvasContainerResized(e);
    }
} });
resizeObserver.observe(document.getElementById("canvas-container"));

onGoButtonClicked();