account_sid = 'ACab07eb0e89e504bc20a61cacc30c8850'
auth_token = '98a9569ce604e8d16173dca5859e78d7'
from_contact = '+12055095159'
to_contact = '+919869548354'
client = Client(account_sid, auth_token)
def sendsms():
  message = client.messages \
                .create(
                     body="Security update: Person spotted on camera ",
                     from_= from_contact ,
                     to= to_contact
                 )

  #print(message.sid)

