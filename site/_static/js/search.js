// List of contractions adapted from Robert MacIntyre's tokenizer.
CONTRACTIONS2 = [
    /(.)('ll|'re|'ve|n't|'s|'m|'d)\b/ig,
    /\b(can)(not)\b/ig,
    /\b(D)('ye)\b/ig,
    /\b(Gim)(me)\b/ig,
    /\b(Gon)(na)\b/ig,
    /\b(Got)(ta)\b/ig,
    /\b(Lem)(me)\b/ig,
    /\b(Mor)('n)\b/ig,
    /\b(T)(is)\b/ig,
    /\b(T)(was)\b/ig,
    /\b(Wan)(na)\b/ig
];
CONTRACTIONS3 = [
    /\b(Whad)(dd)(ya)\b/ig,
    /\b(Wha)(t)(cha)\b/ig
];

tokenize = function(text) {
    $.each(CONTRACTIONS2, function(i, regexp) {
        text = text.replace(regexp,"$1 $2");
    })
    $.each(CONTRACTIONS3, function(i, regexp) {
        text = text.replace(regexp,"$1 $2 $3");
    })

    // Separate most punctuation
    text = text.replace(/([&-]|[\.\!\?])/g, " $1 ");

    // Separate commas if they're followed by space.
    text = text.replace(/(,\s)/," $1");

    // Space out front single quotes if followed by 3 characters
    text = text.replace(/([\'])(\w{3})/g, " $1 $2 ");

    // Separate single quotes if they're followed by a space.
    text = text.replace(/('\s)/," $1");

    // Separate periods that come before newline or end of string.
    text = text.replace(/\. *(\n|$)/," . ");

    // Clean spaces
    text = text.replace(/\s{2,}/, " ").trim()

    return text.split(' ');
}

function sort(object) {
    var sortable = [];
    for (var key in object) {
        sortable.push([key, object[key]]);
    }

    return sortable.sort(function(a, b) {
        return b[1] - a[1];
    });
}


function findRelevantDocuments(query) {
    var queryTokens = tokenize(query.toLowerCase());
    var indeces = [];

    $.each(queryTokens, function (i, token) {
        indeces.push(searchIndex.vocabulary[token]);
    });

    if (indeces.length == 0) {
        console.log("Cannot find documents relevant to the specified query");
        return;
    }

    var rank = {}

    $.each(indeces, function (i, index) {
        $.each(searchIndex.tf.col, function (i, colId) {
            if (index == colId) {
                rowId = searchIndex.tf.row[i]

                if (!rank.hasOwnProperty(rowId)) {
                    rank[rowId] = 0;
                }

                rank[rowId] += searchIndex.tf.data[i] * searchIndex.idf[index];
            }
        });
    });

    var results = [];
    $.each(sort(rank), function (i, data) {
        documentId = data[0];
        documentRank = data[1];
        results.push(searchIndex.documents[documentId]);
    });

    return results;
}

function search(query) {
    var documents = findRelevantDocuments(query);
    var resultsList = $('#results-list .search');

    $('#searching-in-progress').hide();

    if (documents.length == 0) {
        $('#nothing-found').show();
        return;

    } else if (documents.length == 1) {
        $("#number-of-results").text("Found 1 relevant page");
        $('#results').show();

    } else {
        $("#number-of-results").text("Found " + documents.length + " relevant pages");
        $('#results').show();
    }

    $.each(documents, function (i, document) {
        var documentElement = $('<li style="display: list-item;"></li>'),
            link = $('<a />', {text: document.title, href: document.uri}),
            snippet, tag;

        if (document.tag) {
            tag = $('<span />', {text: document.tag, class: "tag"});
            tag.appendTo(documentElement);
        }

        link.appendTo(documentElement);

        if (document.snippet != '') {
            snippet = $('<p/>', {text: document.snippet});
            snippet.appendTo(documentElement);
        }

        documentElement.appendTo(resultsList);
    });
}


$(document).ready(function() {
    var params = $.getQueryParameters();
    if (params.q) {
        var query = params.q[0];
        $('input[name="q"]')[0].value = query;
        search(query);
    }
});
