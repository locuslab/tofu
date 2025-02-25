// Function to handle button clicks
function handleButtonClick(event) {
    // Get all content divs and buttons
    var contents = document.querySelectorAll('.content');
    var buttons = document.querySelectorAll('.button');

    // Remove the "active" class from all buttons and contents
    buttons.forEach(function(btn) {
        btn.classList.remove('active');
    });
    contents.forEach(function(content) {
        content.style.display = 'none';
    });

    // Get the button that was clicked and add "active" class
    var clickedButton = event.target;
    clickedButton.classList.add('active');

    // Determine the corresponding content id
    var contentId = 'content' + clickedButton.id.replace('button', '');
    // Show the corresponding content and add "active" class
    var activeContent = document.getElementById(contentId);
    if (activeContent) {
        activeContent.style.display = 'block';
        activeContent.classList.add('active');
    }
}

// Get all buttons
var buttons = document.querySelectorAll('.button');
// Add click event listeners to all buttons
buttons.forEach(function(button) {
    button.addEventListener('click', handleButtonClick);
});
