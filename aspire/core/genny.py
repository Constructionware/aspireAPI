#  genny.py
#  
#  Copyright 2019 Ian moncrieffe <zellaio600@gmail.com>
#  
# Author:: Ian Moncrieffe (<zellaio600@gmail.com>)
# Copyright:: Copyright (c) 2019 Ian Moncrieffe
# License:: MIT
#
# Licensed under the MIT Licens.
# you may not use this file except in compliance with the License.
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.




from strgen import StringGenerator

class GenerateId:
    tags = dict(
            doc='[h-z5-9]{8:16}',
            app='[a-z0-9]{16:32}',
            key='[a-z0-9]{32:32}',
            job='[a-j0-7]{8:8}',
            user='[0-9]{4:6}',
            item='[a-n1-9]{8:8}',
            code='[a-x2-8]{24:32}'
        )
        
    def genid(self, doc_tag):
        """ 
            Doc Tags: String( doc, app, key, job, user, item, code,task,name)
            UseCase: 
                        >>> import genny
                        >>> from genny import genid
                        >>> from genny import genid as gi
                        
                        >>> id = genny.genid('user')
                        >>> id = genid('user')
                        >>> id = gi('user')
                Yeilds ... U474390
                        ... U77301642
                        ... U1593055
        
        """
        
        if doc_tag == 'user':
            u_id = StringGenerator(str(self.tags[doc_tag])).render(unique=True)
            u_id = 'U{}'.format(u_id)
        else:
            u_id = StringGenerator(str(self.tags[doc_tag])).render(unique=True)
        return u_id
        

    def nameid(self, fn='Jane',ln='Dear',sec=5):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import nameid
                        >>> from genny import nameid as nid
                        
                        >>> id = genny.nameid('Peter','Built',6)
                        >>> id = nameid('Peter','Built',5)
                        >>> id = nid('Peter','Built',4)
                        >>> id = nid() # default false id 
                        
                Yeilds ... PB474390
                        ... PB77301
                        ... PB1593
                        ... JD1951
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        prefix = '{fni}{lni}'.format(fni=fn[0].capitalize(),lni=ln[0].capitalize())
        u_id = StringGenerator(str(code)).render(unique=True)
        u_id = f"{prefix}{u_id}"
        
        return u_id
        

    def shortnameid(self, fn='Jane',ln='Dear',sec=2):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import shortnameid
                        >>> from genny import shortnameid as id
                        
                        >>> id = genny.shortnameid('Peter','Built',2)
                        >>> id = shortnameid('Peter','Built')
                        >>> id = id(p','b',3)
                        >>> id = id() # default false id 
                        
                Yeilds ... PB47
                        ... PB54
                        ... PB69
                        ... JD19
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        prefix = '{fni}{lni}'.format(fni=fn[0].capitalize(),lni=ln[0].capitalize())
        u_id = StringGenerator(str(code)).render(unique=True)
        u_id = '{pre}{id}'.format(pre=prefix,id=u_id)
        
        return u_id



    def eventid(self, event,event_code,sec=8):
        """ 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import eventid
                        >>> from genny import eventid as id
                        
                        >>> id = genny.eventid('Product','LAUNCH',6)
                        >>> id = eventid('Product','LAUNCH',5)
                        >>> id = id('Product', 'LAUNCH',4)
                        
                        
                Yeilds ... PROLAUNCH-884730
                        ... PROLAUNCH-18973
                        ... PROLAUNCH-4631
                        
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        prefix = '{fni}{lni}'.format(fni=event[:3].upper(),lni=event_code)
        u_id = StringGenerator(str(code)).render(unique=True)
        u_id = '{pre}-{id}'.format(pre=prefix,id=u_id)
        
        return u_id



    def shorteventid(self, event,event_code,sec=2):
        """ 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=2.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import shorteventid
                        >>> from genny import shorteventid as id
                        
                        >>> id = genny.shorteventid('Product','LAUNCH',2)
                        >>> id = shorteventid('Product','LAUNCH')
                        >>> id = id('Product', 'LAUNCH',3)
                        
                        
                Yeilds ... PROLAUNCH-88
                        ... PROLAUNCH-90
                        ... PROLAUNCH-461
                        
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        prefix = '{fni}{lni}'.format(fni=event[:3].upper(),lni=event_code)
        u_id = StringGenerator(str(code)).render(unique=True)
        u_id = '{pre}-{id}'.format(pre=prefix,id=u_id)
        
        return u_id
