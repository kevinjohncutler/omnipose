// Select the "show code cell source" button
var button = document.querySelector('.code-cell .celltoolbar .btn.code-cell-toggle');

// Select the first comment in the code cell
var comment = document.querySelector('.code-cell .input_area .cm-comment');

// Change the text of the button to the text of the first comment
if (comment) {
    button.textContent = comment.textContent;
}


window.addEventListener('hashchange', function() {
    // Get the header element
    var header = document.querySelector('.wy-side-nav-search');

    // Set the background color
    header.style.backgroundColor = '#0b750a';

    // Set a delay and then change the color back
    setTimeout(function() {
        header.style.backgroundColor = '';
    }, 3000);
});
