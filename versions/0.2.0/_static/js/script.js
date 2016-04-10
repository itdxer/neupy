$(document).ready(function () {
    $('dl.attribute, dl.method').remove();

    // Bad hack, but it was eas solution.
    $('nav[role="navigation"] a[href$="page1.html"]').each(function () {
        var that = $(this),
            link = that.attr('href');
        that.attr('href', link.substring(0, link.length - 10) + "archive.html");
    });
});
