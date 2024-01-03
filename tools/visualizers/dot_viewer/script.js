
//
// Copyright © 2022 Arm Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//

var canvas = document.getElementById("canvas");

var initialized = false;

var svgRoot = null;
var searchDatabase = [];
var pathConnections = [];

var viewportTop = 0;
var viewportLeft = 0;
var viewportScale = 1.0;
// Right and bottom are updated automatically based on the others, for convenience.
var viewportRight = 0;
var viewportBottom = 0;

var animatedViewportStart;
var animatedViewportEnd;

let worldToViewportTransform = new DOMMatrix();

var mouseDownX;
var mouseDownY;

// A cached version of ctx.getTransform(), used mainly because when setting the transform,
// some precision is lost, which means setTransformIfDiff can't compare.
var cachedCtxTransform = new DOMMatrix();

var scheduledRedrawId = null;

var highlightedElement = null;

// Animating the vieport can be helpful to see where abouts in the graph something you're flying to is located,
// e.g. when following a connection
function startViewportAnimation(endViewportLeft, endViewportTop, endViewportScale) {
    // Animating each variable independently does not result in the expected animation, so animate the centre point instead
    animatedViewportStart = {
        centreX: 0.5 * (viewportLeft + viewportRight),
        centreY: 0.5 * (viewportTop + viewportBottom),
        scale: viewportScale,
        time: performance.now()
    };
    animatedViewportEnd = {
        centreX: endViewportLeft + 0.5 * canvas.width / endViewportScale,
        centreY: endViewportTop + 0.5 * canvas.height / endViewportScale,
        scale: endViewportScale,
    }
    requestRedraw();
}

function onCanvasContainerResized(e) {
    canvas.width = e.contentRect.width;
    canvas.height = e.contentRect.height;

    requestRedraw();
}

function expandBboxPoint(element, point) {
    if (!element.bbox)
    {
        element.bbox = {
            left: point.x,
            top: point.y,
            right: point.x,
            bottom: point.y,
        }
    } else {
        element.bbox = {
            left: Math.min(element.bbox.left, point.x),
            top: Math.min(element.bbox.top, point.y),
            right: Math.max(element.bbox.right, point.x),
            bottom: Math.max(element.bbox.bottom, point.y),
        }
    }
}

function expandBboxRect(element, rect) {
    if (!element.bbox)
    {
        element.bbox = {
            left: rect.left,
            top: rect.top,
            right: rect.right,
            bottom: rect.bottom,
        }
    } else {
        element.bbox = {
            left: Math.min(element.bbox.left, rect.left),
            top: Math.min(element.bbox.top, rect.top),
            right: Math.max(element.bbox.right, rect.right),
            bottom: Math.max(element.bbox.bottom, rect.bottom),
        }
    }
}

function preprocessSvg(e, parentTransform) {
    transform = parentTransform;
    if (e.transform) {
        // Multiply transforms together into a single one, for easier rendering
        e.transform.baseVal.consolidate();
        
        if (e.transform.baseVal.length == 1) {
            let m = e.transform.baseVal.getItem(0).matrix;
            let m2 = new DOMMatrix();
            m2.a = m.a;
            m2.b = m.b;
            m2.c = m.c;
            m2.d = m.d;
            m2.e = m.e;
            m2.f = m.f;    
            transform = parentTransform.multiply(m2);
        }
    }

    // Store world transform for use later
    e.worldTransform = transform;

    // Store world-space bounding box for culling
    if (e.getBBox) {
        let localBbox = e.getBBox();
        let p = new DOMPoint();
        p.x = localBbox.x;
        p.y = localBbox.y;
        let wp1 = transform.transformPoint(p)
        expandBboxPoint(e, wp1);    

        p.x = localBbox.x + localBbox.width;
        p.y = localBbox.y + localBbox.height;
        let wp2 = transform.transformPoint(p)
        expandBboxPoint(e, wp2);    

        // Expand the bbox a little bit to account for the stroke width, this is especially important
        // for vertical/horizontal lines, which would otherwise get culled incorrectly when zoomed quite far in,
        // and also breaks hit-testing
        e.bbox.left -= 1;
        e.bbox.top -= 1;
        e.bbox.right += 1;
        e.bbox.bottom += 1;

        e.characteristicSize = Math.max(e.bbox.right - e.bbox.left, e.bbox.bottom - e.bbox.top);
    }

    if (e instanceof SVGPathElement) {
        // Parse path data and turn into something that can be easily rendered - a Path2D object
        // a list of function calls that do the rendering! 
        let s = e.attributes.d.nodeValue;
        e.canvasPath = new Path2D();

        let commandStrings = s.split(new RegExp('(M|C)')); // By capturing the command letter, it is spliced into the output array
        for (let si = 1; si < commandStrings.length; si += 2) { // Start at 1 to skip the mysterious blank entry at the start
            let commandName = commandStrings[si];
            let commandArgs = commandStrings[si+1];

            if (commandName == "M") {
                let xy = commandArgs.split(",");
                let x = parseFloat(xy[0]);
                let y = parseFloat(xy[1]);
                e.canvasPath.moveTo(x, y);
            } else if (commandName == "C") {
                let points = commandArgs.split(" ");
                for (var i in points) {
                    points[i] = points[i].split(",");
                    points[i][0] = parseFloat(points[i][0]);
                    points[i][1] = parseFloat(points[i][1]);
                }
                for (let pi = 0; pi < points.length; pi += 3) {
                    let c1x = points[pi][0];
                    let c1y = points[pi][1];
                    let c2x = points[pi+1][0];
                    let c2y = points[pi+1][1];
                    let dx = points[pi+2][0];
                    let dy = points[pi+2][1];
                    e.canvasPath.bezierCurveTo(c1x, c1y, c2x, c2y, dx, dy);
                }
            }
            else {
                console.log("Unknown path command: " + commandName);
            }
        }

        // Store the start and end of each path, for quick following of connections
        var a = transform.transformPoint(e.getPointAtLength(0));
        var b = transform.transformPoint(e.getPointAtLength(e.getTotalLength()));
        // Store both directions, so can jump both ways easily
        pathConnections.push({ start: a, end: b, element: e});
        pathConnections.push({ start: b, end: a, element: e});
    } else if (e instanceof SVGTextElement) {
        // Override characteristicSize to be based on the size of one character - having a large block
        // of text zoomed out is not very useful to render (can't read it anyway), so prefer to LOD this.
        e.characteristicSize = 3 * parseFloat(e.attributes["font-size"].value);
        // Store text for search
        searchDatabase.push({
            text: e.childNodes[0].nodeValue,
            element: e
        });
    }

    for (let child of e.children) {
        preprocessSvg(child, transform);
    }
}

function focusWorldRect(rect, paddingFactor, animate=false) {
    w = (rect.right - rect.left) * (1 + 2 * paddingFactor);
    h = (rect.bottom - rect.top) * (1 + 2 * paddingFactor);
    l = rect.left - (rect.right - rect.left) * paddingFactor;
    t = rect.top - (rect.bottom - rect.top) * paddingFactor;
    r = rect.right + (rect.right - rect.left) * paddingFactor;
    b = rect.bottom + (rect.bottom - rect.top) * paddingFactor;

    var scaleW = canvas.width / w;
    var scaleH = canvas.height / h;
    var newScale;
    var newLeft;
    var newTop;
    if (scaleW < scaleH) {
        newScale = scaleW;
        newLeft = l;
        newTop = 0.5 * (t + b) - 0.5 * canvas.height / newScale;
    } else {
        newScale = scaleH;
        newLeft = 0.5 * (l + r) - 0.5 * canvas.width / newScale;
        newTop = t;        
    }
                 
    if (animate) {
        startViewportAnimation(newLeft, newTop, newScale);
    } else {
        viewportLeft = newLeft;
        viewportTop = newTop;
        viewportScale = newScale;
        requestRedraw();
    }
}

function focusWorldPoint(x, y) {
    var newLeft = x - 0.5 * canvas.width / viewportScale;
    var newTop = y - 0.5 * canvas.height / viewportScale;
    startViewportAnimation(newLeft, newTop, viewportScale);
}

function resetViewport() {
    focusWorldRect(svgRoot.bbox, 0.1);
}

function initialize(svgSource) {
    var tempElement = document.createElement("html");
    tempElement.innerHTML = svgSource;
    svgRoot = tempElement.getElementsByTagName("svg")[0];

    document.getElementById("svg-source-container").appendChild(svgRoot);
    preprocessSvg(svgRoot, new DOMMatrix());

    // Once processed, we can hide the original SVG (it needs to be displayed in order for getBBox to work!)
    document.getElementById("svg-source-container").style.display = "none";

    resetViewport();

    document.getElementById("canvas").focus();

    initialized = true;
    document.getElementById("loading").style.display = "none";
    requestRedraw();
}

var drawStats = {
    numDrawn: 0,
    numCulled: 0,
    numLods: 0
}

function matricesEqual(a, b) {
    return a.a == b.a && 
    a.b == b.b && 
    a.c == b.c && 
    a.d == b.d && 
    a.e == b.e && 
    a.f == b.f;
}

function setTransformIfDiff(ctx, newTransform) {
    // Note we compare against cachedCtxTransform, not ctz.getTransform(), because the latter
    // has some precision loss when a transform is stored and retrieved
    if (!matricesEqual(cachedCtxTransform, newTransform)) {
        ctx.setTransform(newTransform);
        cachedCtxTransform = newTransform;
    }
}

function draw(ctx, e, worldToViewportTransform) {
    // Cull this element and all its descendants if its bbox is outside the viewport
    if (e.bbox && (e.bbox.right < viewportLeft || e.bbox.left > viewportRight || e.bbox.top > viewportBottom || e.bbox.bottom < viewportTop)) {
        ++drawStats.numCulled;
        return;
    }

    // Cull this element and all its descendants if its bbox is too small (to avoid rendering small details when zoomed out)
    if (e.characteristicSize && e.characteristicSize * viewportScale < 2) {
        ++drawStats.numCulled;
        return;
    }

    // Replace this element and all its descendants with a simple LOD (black box) if its bbox is too small (to avoid rendering small details when zoomed out)
    // Draw something to indicate that there is something here, zoomed out a bit
    if (e.characteristicSize && e.characteristicSize * viewportScale < 10) {
        ctx.fillStyle =  (e == highlightedElement) ? "yellow" : "grey";
        setTransformIfDiff(ctx, worldToViewportTransform);
        ctx.fillRect(e.bbox.left, e.bbox.top, e.bbox.right - e.bbox.left, e.bbox.bottom - e.bbox.top);
        ++drawStats.numLods;
        return;
    }

    setTransformIfDiff(ctx, worldToViewportTransform.multiply(e.worldTransform));

    if (e.attributes.stroke) {
        ctx.strokeStyle = e.attributes.stroke.value;
    }
    if (e.attributes["stroke-dasharray"]) {
        var array = []
        for (let p of e.attributes["stroke-dasharray"].value.split(",")) {
            array.push(parseFloat(p));
        }
        ctx.setLineDash(array);
    } else {
        ctx.setLineDash([]);  
    }

    if (e instanceof SVGTextElement) {
        if (e.attributes["text-anchor"].value == "middle") {
            ctx.textAlign = "center"
        } else {
            ctx.textAlign = e.attributes["text-anchor"].value;
        }
        ctx.font = e.attributes["font-size"].value + "px " + e.attributes["font-family"].value;
        ctx.fillStyle = "black";
        ctx.fillText(e.childNodes[0].nodeValue, e.x.baseVal.getItem(0).value, e.y.baseVal.getItem(0).value);
    }
    else if (e instanceof SVGPolygonElement || e instanceof SVGPolylineElement) {
        ctx.beginPath();
        ctx.moveTo(e.points.getItem(0).x, e.points.getItem(0).y);
        for (let i = 1; i < e.points.length; ++i) {
            ctx.lineTo(e.points.getItem(i).x, e.points.getItem(i).y);           
        }
        ctx.closePath();
        if (e.attributes.stroke.value != "none") {
            ctx.stroke();
        }
        // Note that we don't fill, even if specified in the svg, as this results in a huge white rectangle being 
        // drawn which covers up the highlighting. Fill doesn't seem to be important for looking correct, so we ignore it.
    }
    else if (e instanceof SVGEllipseElement) {
        ctx.beginPath();
        ctx.ellipse(e.cx.baseVal.value, e.cy.baseVal.value, e.rx.baseVal.value, e.ry.baseVal.value, 0, 0, 2 * Math.PI);
        if (e.attributes.stroke.value != "none") {
            ctx.stroke();
        }
        // Note that we don't fill, even if specified in the svg, as this results in a huge white rectangle being 
        // drawn which covers up the highlighting. Fill doesn't seem to be important for looking correct, so we ignore it.
    }    
    else if (e instanceof SVGPathElement) {
        // There isn't easy access to the path data to be rendered, so we preprocess this into a Path2D object that can be easily drawn

        // Highlight paths specially compared to everything else, because they can be very thin and so the bbox highlighting
        // doesn't work so well for them
        if (e == highlightedElement) {
            ctx.lineWidth = 10;
            ctx.strokeStyle = "yellow";
            ctx.stroke(e.canvasPath);
        }

        if (e.attributes.stroke.value != "none") {
            ctx.stroke(e.canvasPath);
        }
        ctx.lineWidth = 1; // Restore
        // Note that we don't fill, even if specified in the svg, as this results in a huge white rectangle being 
        // drawn which covers up the highlighting. Fill doesn't seem to be important for looking correct, so we ignore it.
    }

    ++drawStats.numDrawn;

    for (let child of e.children) {
        draw(ctx, child, worldToViewportTransform);
    }
}

// Rather than drawing immediately, we use requestAnimationFrame for smoother animation when dragging etc.
function requestRedraw() {
    if (scheduledRedrawId) {
        return; // Already scheduled a redraw, nothing to do
    }
    scheduledRedrawId = window.requestAnimationFrame(redrawImpl);
}

function redrawImpl() {
    scheduledRedrawId = null;
    if (!initialized) {
        return;
    }

    var startTime = performance.now();

    // Update animated viewport, based on the current time 
    // (note that the time provided to us by the browser (this function's first argument), seems to back in time for the first frame!
    if (animatedViewportEnd) {
        var t = Math.min((startTime - animatedViewportStart.time) / 500.0, 1.0); // Hardcoded animation length
        // Interpolate in a way that makes the animation nice
        viewportScale = animatedViewportStart.scale * (1-t) + animatedViewportEnd.scale * t;
        var d = (1 / animatedViewportEnd.scale - 1 / animatedViewportStart.scale);
        var p = Math.abs(d) < 0.01 ? t : (1 / viewportScale - 1 / animatedViewportStart.scale) / d;
        var centreX = animatedViewportStart.centreX * (1-p) + animatedViewportEnd.centreX * p;
        var centreY = animatedViewportStart.centreY * (1-p) + animatedViewportEnd.centreY * p;
        viewportLeft = centreX - 0.5 * canvas.width / viewportScale;
        viewportTop = centreY - 0.5 * canvas.height / viewportScale;

        if (t < 1) {
            // Animation not finished, schedule another redraw
            requestRedraw();
        } else {
            // Animation finished
            animatedViewportStart = null;
            animatedViewportEnd = null;
        }
    }

    var ctx = canvas.getContext("2d");

    viewportRight = viewportLeft + canvas.width / viewportScale;
    viewportBottom = viewportTop + canvas.height / viewportScale;

    ctx.textBaseline = "alphabetic";
    setTransformIfDiff(ctx, new DOMMatrix());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    worldToViewportTransform = new DOMMatrix();
    worldToViewportTransform.a = viewportScale;
    worldToViewportTransform.d = viewportScale;
    worldToViewportTransform.e = -viewportLeft * viewportScale;
    worldToViewportTransform.f = -viewportTop * viewportScale;

    // If an element is highlighted, draw a bright box around it, behind everything
    // Path elements are handled specially though (see their drawing code)
    if (highlightedElement && !(highlightedElement instanceof SVGPathElement)) {
        ctx.fillStyle = "yellow";
        setTransformIfDiff(ctx, worldToViewportTransform);
        ctx.fillRect(highlightedElement.bbox.left, highlightedElement.bbox.top, highlightedElement.bbox.right - highlightedElement.bbox.left, highlightedElement.bbox.bottom - highlightedElement.bbox.top);
    }

    drawStats.numDrawn = 0;
    drawStats.numCulled = 0;
    drawStats.numLods = 0;
    draw(ctx, svgRoot, worldToViewportTransform)

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.font = "12px Arial";
    ctx.textBaseline = "top";
    ctx.textAlign = "start"
    ctx.fillStyle = "black";
    ctx.fillText("Drawn:  " + drawStats.numDrawn + ", Culled: " + drawStats.numCulled + ", Lods: " + drawStats.numLods, 10, 10);
 
    var endTime = performance.now();
    ctx.fillText("Draw time:  " + (endTime - startTime).toString(), 10, 25);
}

function onCanvasMouseDown(event) {
    if (event.button == 0) { // Left-click
        mouseDownX = event.offsetX;
        mouseDownY = event.offsetY;
    } else if (event.button == 1) { // Middle-click
        resetViewport();
    }
}

function onCanvasMouseMove(event) {
    if (event.buttons == 1) { // Left-click
        if (Math.abs(event.offsetX - mouseDownX) > 3 || Math.abs(event.offsetY - mouseDownY) > 3) {
            viewportLeft -= event.movementX / viewportScale;
            viewportTop -= event.movementY / viewportScale;
            requestRedraw();    
        }
    }
}

function hitTest(e, x, y, hits) {
    if (e.bbox && x > e.bbox.left && x < e.bbox.right && y > e.bbox.top && y < e.bbox.bottom) {
        // Handle path elements differently, because their bbox might be way bigger than what we want,
        // especialy if the path is diagonal, so do an additional more accurate test
        if (e instanceof SVGPathElement) {
            // Convert the point to local coords
            var p = e.worldTransform.inverse().transformPoint(new DOMPoint(x, y)); 
            var ctx = canvas.getContext("2d");           
            ctx.lineWidth = 5; // Pretend lines are wider than they actually are, so that hit-testing has a bit of leeway
            if (ctx.isPointInStroke(e.canvasPath, p.x, p.y)) {
                hits.push(e);
            }
            ctx.lineWidth = 1; // restore
        } else {
            hits.push(e);
        }
    }

    for (let child of e.children) {
        hitTest(child, x, y, hits);
    }
}

function getElementFromCanvasPos(x, y) {
    var worldX = x / viewportScale + viewportLeft;
    var worldY = y / viewportScale + viewportTop;

    hits = [];
    hitTest(svgRoot, worldX, worldY, hits);
    // Return the deepest (valid) element in the tree, which should be the smallest (most precise)
    for (var i = hits.length - 1; i >= 0; --i) {
        if (hits[i].tagName == "g") {
            continue; // Don't select <g> elements, as these don't correspond to anything really, and it messes up selection of edges
        }
        return hits[i];
    }
    return null;
}

function onCanvasMouseUp(event) {
    if (event.button == 0) { // Left-click
        if (Math.abs(event.offsetX - mouseDownX) < 3 && Math.abs(event.offsetY - mouseDownY) < 3) {
            if (event.altKey) {
                // Find the closest end of a path, and jump the view to the other end of the connection
                var worldX = event.offsetX / viewportScale + viewportLeft;
                var worldY = event.offsetY / viewportScale + viewportTop;
                var closestDistSq = 10e10;
                var connection = null;
                for (var c of pathConnections) {
                    var distSq = (c.start.x - worldX) * (c.start.x - worldX) + (c.start.y - worldY) * (c.start.y - worldY);
                    if (distSq < closestDistSq) {
                        closestDistSq = distSq;
                        connection = c;
                    }
                }
                var dist = Math.sqrt(closestDistSq);
                if (dist < (100 / viewportScale)) {
                    highlightedElement = connection.element;
                    focusWorldPoint(connection.end.x, connection.end.y);
                }
            } else {
                var e = getElementFromCanvasPos(event.offsetX, event.offsetY);
                highlightedElement = e;
                requestRedraw();
            }
        }
    }
}

function onKeyDown(event) {
    // This is a global listener, so avoid intercepting keys that could be typing commands (e.g. in the search box)
    if (event.target.tagName != "INPUT" && event.target.tagName != "TEXTAREA") {
        if (event.code == "KeyW") {
            viewportTop -= 100 / viewportScale;
            requestRedraw();
        } else if (event.code == "KeyS") {
            viewportTop += 100 / viewportScale;
            requestRedraw();
        } else if (event.code == "KeyA") {
            viewportLeft -= 100 / viewportScale;
            requestRedraw();
        } else if (event.code == "KeyD") {
            viewportLeft += 100 / viewportScale;
            requestRedraw();
        } else if (event.code == "Home") {
            resetViewport();
        } else if (event.code == "KeyF" && !event.ctrlKey) {
            if (highlightedElement) {
                focusWorldRect(highlightedElement.bbox, 0.5, true);            
            }
        } 
    }
    if (event.code == "KeyF" && event.ctrlKey) {
        document.getElementById("search-query").focus();
        event.preventDefault(); // Prevnt browser find
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

    requestRedraw();
}

// Recommendation from https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

function refreshSearch() {
    document.getElementById("search-results").innerHTML = "";

    var query = document.getElementById("search-query").value;
    document.getElementById("search-query").classList.remove("error");
    if (query.trim().length == 0) {
        return;
    }

    try {
        var regex = new RegExp(document.getElementById("search-regex").checked ? query : escapeRegExp(query),
            document.getElementById("search-case-sensitive").checked ? "g" : "ig");
    } catch (error) {
        document.getElementById("search-query").classList.add("error");
        
        let div = document.createElement("div");
        div.classList.add("search-result");
        div.innerText = error;
        document.getElementById("search-results").appendChild(div);

        return;
   }

    let num_results = 0;
    for (let entry of searchDatabase) {
        for (let match of entry.text.matchAll(regex))
        {
            let div = document.createElement("div");
            div.classList.add("search-result");
            div.searchResult = entry;
            div.addEventListener("click", onSearchResultClick);

            let before = document.createTextNode(entry.text.substr(0, match.index));
            let middle = document.createElement("mark");
            middle.innerText = entry.text.substr(match.index, match[0].length);
            let end = document.createTextNode(entry.text.substr(match.index + match[0].length));
            div.appendChild(before);
            div.appendChild(middle);
            div.appendChild(end);

            document.getElementById("search-results").appendChild(div);

            ++num_results;
        }
        if (num_results > 100) {
            let div = document.createElement("div");
            div.classList.add("search-result");
            div.innerText = "Too many results, stopping here!";
            document.getElementById("search-results").appendChild(div);
            break;
        }
    }

    if (num_results == 0) {
        let div = document.createElement("div");
        div.classList.add("search-result");
        div.innerText = "No results!";
        document.getElementById("search-results").appendChild(div);
    }
}

function onSearchResultClick(e) {
    let searchResult = e.currentTarget.searchResult;
    highlightedElement = searchResult.element;
    focusWorldRect(searchResult.element.bbox, 2.0, true);
}

document.getElementById("canvas").addEventListener("mousedown", onCanvasMouseDown);
document.getElementById("canvas").addEventListener("mousemove", onCanvasMouseMove);
document.getElementById("canvas").addEventListener("mouseup", onCanvasMouseUp);
document.getElementById("canvas").addEventListener("wheel", onCanvasWheel);
document.addEventListener("keydown", onKeyDown);
document.getElementById("search-query").addEventListener("input", refreshSearch);
document.getElementById("search-case-sensitive").addEventListener("change", refreshSearch);
document.getElementById("search-regex").addEventListener("change", refreshSearch);

var resizeObserver = new ResizeObserver(entries => { for (let e of entries) {
    if (e.target.id == "canvas-container") {
        onCanvasContainerResized(e);
    }
} });
resizeObserver.observe(document.getElementById("canvas-container"));

// Delay initialising stuff, so that the above resize handler will get called first
window.setTimeout(function() {
    fetch('Source.svg')
    .then((response) => {
        document.title = response.headers.get("X-Dot-Filename");
        return response.text();
    })
    .then((svgSource) => initialize(svgSource));  
}, 50);
