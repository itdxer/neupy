from neupy.helpers.logs import Verbose


__all__ = ('twilio_sms',)


def twilio_sms(account_id, token, to_phone, from_phone, verbose=True):
    """
    Send SMS via Twilio service.

    Parameters
    ----------
    account_id : str
        Twilio account ID.
    token : str
        Twilio account token.
    to_phone : str
        SMS receiver phone number.
    from_phone : str
        SMS sender phone number.
    verbose : bool
        Logging verbose. Defaults to ``True``.

    Returns
    -------
    func
        Retunr function which take one text message argument and send it
        via Twilio API.
    """
    verbose = Verbose(verbose=verbose)

    try:
        import twilio
    except ImportError:
        raise ImportError("Install `twilio` library.")

    def send_message(text_message):
        formated_message = "Send SMS with text: '{}'".format(text_message)
        verbose.message("SMS", formated_message)

        client = twilio.rest.TwilioRestClient(account_id, token)
        message = client.messages.create(body=text_message, to=to_phone,
                                         from_=from_phone)
        return message
    return send_message
