import sys

from neupy.helpers.logs import Verbose


__all__ = ('twilio_sms',)


def twilio_sms(account_id, token, to_phone, from_phone, verbose=True):
    """ Send SMS via Twilio service.

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
    logs = Verbose(verbose=verbose).logs

    try:
        import twilio
    except ImportError:
        logs.error("Install twilio module before use this function. Command:")
        logs.simple("pip install twilio")
        sys.exit()

    def send_message(text_message):
        logs.log("Message", "Send SMS with text: '{}'".format(text_message))

        client = twilio.rest.TwilioRestClient(account_id, token)
        message = client.messages.create(body=text_message, to=to_phone,
                                         from_=from_phone)
        return message
    return send_message
