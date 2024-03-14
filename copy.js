document.addEventListener("DOMContentLoaded", function () {
    var copyButton = document.querySelectorAll('.btn-copy');
    copyButton.forEach(function (button) {
        button.addEventListener('click', function () {
            var codeBlock = button.parentNode.querySelector('pre');
            var range = document.createRange();
            range.selectNode(codeBlock);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.execCommand('copy');
            window.getSelection().removeAllRanges();
            button.innerHTML = 'Copied!';
            setTimeout(function () {
                button.innerHTML = 'Copy';
            }, 2000);
        });
    });
});
